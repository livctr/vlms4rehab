import React, { useState, useEffect, useRef } from 'react';
import './App.css';

const NODE = process.env.NODE || "localhost";
const PORT = process.env.PORT || 9172;
const API_URL = `http://${NODE}:${PORT}/api`;

function App() {
  const [data, setData] = useState(null);
  const [selectedLeftHandIdx, setSelectedLeftHandIdx] = useState(-1);
  const [selectedRightHandIdx, setSelectedRightHandIdx] = useState(-1);
  const [scale, setScale] = useState(1);
  const imageRef = useRef(null);

  // Fetch data from /api/data with a given navigation parameter
  const fetchData = (navigation = 'current') => {
    fetch(`${API_URL}/data?navigation=${navigation}`)
      .then((res) => res.json())
      .then((json) => {
        // Get dimensions of the image before setting state
        if (json && json.frame) {
          const img = new Image();
          img.onload = () => {
            const dataWithDimensions = {
              ...json,
              original_width: img.width,
              original_height: img.height
            };
            setData(dataWithDimensions);
            setSelectedLeftHandIdx(-1); // reset selection on new data
            setSelectedRightHandIdx(-1); // reset selection on new data
            
            // Reset the imageRef current if it exists
            if (imageRef.current) {
              imageRef.current.src = `data:image/png;base64,${json.frame}`;
            }
          };
          img.src = `data:image/png;base64,${json.frame}`;
        } else {
          setData(json);
          setSelectedLeftHandIdx(-1); // reset selection on new data
          setSelectedRightHandIdx(-1); // reset selection on new data
        }
      })
      .catch((err) => console.error(err));
  };

  // Fetch the initial data when the component mounts
  useEffect(() => {
    fetchData();
  }, []);

  // Send selection info to /api/select_hands
  const postSelection = () => {
    const { frame, ...dataWithoutFrame } = data;
    const payload = {
      ...dataWithoutFrame,
      left_hand_idx: selectedLeftHandIdx,
      right_hand_idx: selectedRightHandIdx,
    };
    fetch(`${API_URL}/select_hands`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    })
      .then((res) => res.json())
      .then((json) => {
        if (json.success) {
          // Optionally load the next frame after successful submission
          fetchData();
        }
      })
      .catch((err) => console.error(err));
  };

  // Handler for "Select" button
  const handleSelect = () => {
    postSelection();
  };

  // Handler for "No hands" button
  const handleNoHands = () => {
    setSelectedLeftHandIdx(-1);
    setSelectedRightHandIdx(-1);
    postSelection();
  };

  // Navigation handlers
  const handlePrev = () => {
    fetchData('previous');
  };

  const handleNext = () => {
    fetchData('next');
  };

  // Save handler
  const handleSave = () => {
    fetch(`${API_URL}/save`, { method: 'POST' })
      .then((res) => res.json())
      .then((json) => {
        console.log('Save response:', json);
      })
      .catch((err) => console.error(err));
  };

  // When a bounding box is clicked, update the selected index
  const handleBoxClick = (index) => {
    // If already selected as left hand, switch to right hand
    if (index === selectedLeftHandIdx) {
      setSelectedLeftHandIdx(-1);
      setSelectedRightHandIdx(index);
    }
    // If already selected as right hand, switch to left hand
    else if (index === selectedRightHandIdx) {
      setSelectedRightHandIdx(-1);
      setSelectedLeftHandIdx(index);
    }
    // If not selected, toggle between left and right hand selection
    else {
      // If no hands are selected, default to left hand
      if (selectedLeftHandIdx === -1 && selectedRightHandIdx === -1) {
        setSelectedLeftHandIdx(index);
      }
      // If only left hand is selected, allow selecting right hand
      else if (selectedLeftHandIdx !== -1 && selectedRightHandIdx === -1) {
        setSelectedRightHandIdx(index);
      }
      // If only right hand is selected, allow selecting left hand
      else if (selectedLeftHandIdx === -1 && selectedRightHandIdx !== -1) {
        setSelectedLeftHandIdx(index);
      }
      // If both hands are selected, replace the one that was clicked last
      else {
        // This case shouldn't happen due to the above conditions, but just in case
        setSelectedLeftHandIdx(index);
        setSelectedRightHandIdx(-1);
      }
    }
  };

  // Adjust scale factor when the image loads. Uses the displayed width vs. original width.
  const handleImageLoad = (event) => {
    if (data && data.original_width) {
      const displayedWidth = event.target.clientWidth;
      const computedScale = displayedWidth / data.original_width;
      setScale(computedScale);
    }
  };

  // Get current selection step
  const getSelectionStep = () => {
    if (selectedLeftHandIdx === -1 && selectedRightHandIdx === -1) {
      return 1;
    } else if (selectedLeftHandIdx !== -1 && selectedRightHandIdx === -1) {
      return 2;
    } else if (selectedLeftHandIdx === -1 && selectedRightHandIdx !== -1) {
      return 3;
    }
    return 4;
  };

  // Get instructions based on current step
  const getInstructions = () => {
    const step = getSelectionStep();
    switch (step) {
      case 1:
        return "Step 1: Click on a box to select either the LEFT hand (blue) or RIGHT hand (green)";
      case 2:
        return "Step 2: You've selected the LEFT hand. Click another box to select the RIGHT hand, or click 'Select' to confirm just the left hand";
      case 3:
        return "Step 3: You've selected the RIGHT hand. Click another box to select the LEFT hand, or click 'Select' to confirm just the right hand";
      case 4:
        return "Step 4: Review your selections. Click 'Select' to confirm or click a box to change selection";
      default:
        return "";
    }
  };

  // Render the bounding boxes using the computed scale factor
  const renderBoundingBoxes = () => {
    if (!data || !data.hand_bboxes) return null;
    return data.hand_bboxes.map((bbox, index) => {
      const [x1, y1, x2, y2] = bbox;
      // Adjust coordinates and dimensions by the scale factor
      const adjustedX = x1 * scale;
      const adjustedY = y1 * scale;
      const adjustedWidth = (x2 - x1) * scale;
      const adjustedHeight = (y2 - y1) * scale;
      
      // Determine box style based on selection
      let borderColor = 'red';
      let borderWidth = '1px';
      let label = '';
      
      if (index === selectedLeftHandIdx) {
        borderColor = 'blue';
        borderWidth = '3px';
        label = 'L';
      } else if (index === selectedRightHandIdx) {
        borderColor = 'green';
        borderWidth = '3px';
        label = 'R';
      }
      
      return (
        <div
          key={index}
          onClick={() => handleBoxClick(index)}
          style={{
            position: 'absolute',
            left: adjustedX,
            top: adjustedY,
            width: adjustedWidth,
            height: adjustedHeight,
            border: `${borderWidth} solid ${borderColor}`,
            boxSizing: 'border-box',
            cursor: 'pointer',
          }}
        >
          {label && (
            <div style={{
              position: 'absolute',
              top: -20,
              left: 0,
              backgroundColor: borderColor,
              color: 'white',
              padding: '2px 6px',
              borderRadius: '3px',
              fontSize: '14px',
              fontWeight: 'bold'
            }}>
              {label}
            </div>
          )}
        </div>
      );
    });
  };

  return (
    <div className="container">
      {data ? (
        <div className="annotation-ui">
          <h1>Hand Selection</h1>
          <h5>{data.num_frames_done}/{data.num_frames_needed} annotations completed.</h5>
          
          <div className="instructions" style={{
            backgroundColor: '#f0f0f0',
            padding: '15px',
            borderRadius: '5px',
            marginBottom: '20px',
            border: '1px solid #ddd'
          }}>
            <h3 style={{ margin: '0 0 10px 0' }}>{getInstructions()}</h3>
            <div style={{ display: 'flex', gap: '20px', marginTop: '10px' }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: '5px' }}>
                <div style={{ width: '20px', height: '20px', backgroundColor: 'blue', border: '2px solid blue' }}></div>
                <span>Left Hand</span>
              </div>
              <div style={{ display: 'flex', alignItems: 'center', gap: '5px' }}>
                <div style={{ width: '20px', height: '20px', backgroundColor: 'green', border: '2px solid green' }}></div>
                <span>Right Hand</span>
              </div>
              <div style={{ display: 'flex', alignItems: 'center', gap: '5px' }}>
                <div style={{ width: '20px', height: '20px', border: '1px solid red' }}></div>
                <span>Unselected</span>
              </div>
            </div>
          </div>

          <div className="top-buttons">
            <button onClick={handleSelect}>Select</button>
            <button onClick={handleNoHands}>No Selection</button>
            <button onClick={handlePrev}>Prev</button>
            <button onClick={handleNext}>Next</button>
            <button onClick={handleSave}>Save</button>
          </div>
          <div className="image-container" style={{ position: 'relative', display: 'inline-block' }}>
            <img
              ref={imageRef}
              src={`data:image/png;base64,${data.frame}`}
              alt="Frame"
              onLoad={handleImageLoad}
              style={{ display: 'block', maxWidth: '100%' }}
            />
            {renderBoundingBoxes()}
          </div>
          <div className="info-panel">
            <p>
              Image size: {data.original_width || "Loading..."} x {data.original_height || "Loading..."} px
            </p>
            <p>Bounding boxes:</p>
            <ul>
              {data.hand_bboxes && data.hand_bboxes.map((bbox, index) => (
                <li 
                  key={index} 
                  style={{ 
                    color: index === selectedLeftHandIdx ? 'blue' : 
                           index === selectedRightHandIdx ? 'green' : 'black',
                    cursor: 'pointer',
                    padding: '5px',
                    backgroundColor: index === selectedLeftHandIdx ? '#e6f3ff' :
                                   index === selectedRightHandIdx ? '#e6ffe6' : 'transparent',
                    borderRadius: '3px'
                  }}
                  onClick={() => handleBoxClick(index)}
                >
                  Box {index + 1}: [{bbox[0]}, {bbox[1]}, {bbox[2]}, {bbox[3]}]
                  {index === selectedLeftHandIdx && " (Left Hand)"}
                  {index === selectedRightHandIdx && " (Right Hand)"}
                </li>
              ))}
              {(!data.hand_bboxes || data.hand_bboxes.length === 0) && <li>No boxes detected</li>}
            </ul>
          </div>
        </div>
      ) : (
        <div>Loading...</div>
      )}
    </div>
  );
}

export default App;
