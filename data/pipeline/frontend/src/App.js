import React, { useState, useEffect, useRef } from 'react';
import './App.css';

const NODE = process.env.NODE || "localhost";
const PORT = process.env.PORT || 9172;
const API_URL = `http://${NODE}:${PORT}/api`;

function App() {
  const [data, setData] = useState(null);
  const [selectedBoxIdx, setSelectedBoxIdx] = useState(-1);
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
            setSelectedBoxIdx(-1); // reset selection on new data
            
            // Reset the imageRef current if it exists
            if (imageRef.current) {
              imageRef.current.src = `data:image/png;base64,${json.frame}`;
            }
          };
          img.src = `data:image/png;base64,${json.frame}`;
        } else {
          setData(json);
          setSelectedBoxIdx(-1); // reset selection on new data
        }
      })
      .catch((err) => console.error(err));
  };

  // Fetch the initial data when the component mounts
  useEffect(() => {
    fetchData();
  }, []);

  // Send selection info to /api/select_hand_bbox
  const postSelection = (boxIdx) => {
    const { frame, ...dataWithoutFrame } = data;
    const payload = {
      ...dataWithoutFrame,
      selected_hand_bbox_idx: boxIdx,
    };
    fetch(`${API_URL}/select_hand_bbox`, {
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
    if (selectedBoxIdx === -1) return; // do nothing if no box is selected
    postSelection(selectedBoxIdx);
  };

  // Handler for "No hands" button
  const handleNoHands = () => {
    setSelectedBoxIdx(-1);
    postSelection(-1);
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
    setSelectedBoxIdx(index);
  };

  // Adjust scale factor when the image loads. Uses the displayed width vs. original width.
  const handleImageLoad = (event) => {
    if (data && data.original_width) {
      const displayedWidth = event.target.clientWidth;
      const computedScale = displayedWidth / data.original_width;
      setScale(computedScale);
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
      const isSelected = selectedBoxIdx === index;
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
            border: isSelected ? '3px solid red' : '1px solid red',
            boxSizing: 'border-box',
            cursor: 'pointer',
          }}
        ></div>
      );
    });
  };

  return (
    <div className="container">
      {data ? (
        <div className="annotation-ui">
          <h1>Select the patient's {data.handedness || ""} hand.</h1>
          <h5>{data.num_frames_done}/{data.num_frames_needed} annotations completed.</h5>
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
                <li key={index} style={{ color: selectedBoxIdx === index ? 'red' : 'black' }}>
                  Box {index + 1}: [{bbox[0]}, {bbox[1]}, {bbox[2]}, {bbox[3]}]
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
