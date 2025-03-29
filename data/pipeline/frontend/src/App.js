import React, { useEffect, useState } from "react";
import "./App.css";


const NODE = process.env.NODE || "localhost";
const PORT = process.env.PORT || 9172;
const API_URL = `http://${NODE}:${PORT}/api`;


function App() {
  const [data, setData] = useState(null);
  const [selectedBox, setSelectedBox] = useState(null);

  useEffect(() => {
    fetch(`${API_URL}/data`)
      .then((res) => res.json())
      .then((data) => setData(data))
      .catch(console.error);
  }, []);

  const handleBoxClick = (box) => {
    setSelectedBox(box);
  };

  const handleSubmit = () => {
    if (!selectedBox) {
      alert("Please select a bounding box first.");
      return;
    }
    fetch(`${API_URL}/submit`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ selected_box: selectedBox }),
    })
      .then((res) => res.json())
      .then((response) => {
        console.log(response);
      })
      .catch(console.error);
  };

  if (!data) return <div>Loading...</div>;

  const { image, bounding_boxes, prompt } = data;

  return (
    <div className="container">
      <h2>{prompt}</h2>
      <h2>{prompt}</h2>
      <h2>{prompt}</h2>
      <div className="image-container">
        <img src={`data:image/png;base64,${image}`} alt="Patient scan" />
        {bounding_boxes.map((box) => (
          <div
            key={box.id}
            className={`bounding-box ${selectedBox && selectedBox.id === box.id ? "selected" : ""}`}
            style={{
              position: "absolute",
              left: `${box.x}px`,
              top: `${box.y}px`,
              width: `${box.width}px`,
              height: `${box.height}px`,
              border: "2px solid red",
              cursor: "pointer"
            }}
            onClick={() => handleBoxClick(box)}
          />
        ))}
      </div>
      <button onClick={handleSubmit} className="done-button">Done</button>
    </div>
  );
}

export default App;
