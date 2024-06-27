"use client";
import { useState, useEffect, useRef } from "react";

const LINE_OFFSET = 6;
const ANCHOR_SIZE = 2;

export default function ObjectDetection() {
  {
    const canvasRef = useRef(null);
    const image_path =
      "https://miro.medium.com/v2/resize:fit:1400/1*v0Bm-HQxWtpbQ0Yq463uqw.jpeg";
    const [rectangles, setRectangles] = useState([]);
    const [labelCounts, setLabelCounts] = useState({});
    const [mouseDown, setMouseDown] = useState(false);
    const [currentRect, setCurrentRect] = useState(null);
    const [clickedArea, setClickedArea] = useState({ index: -1, pos: "o" });
    const [label, setLabel] = useState("");

    // finds where on the rectangle the click happened
    function findCurrentArea(x, y) {
      for (var i = 0; i < rectangles.length; i++) {
        var rectangle = rectangles[i];
        var xCenter = rectangle.x1 + (rectangle.x2 - rectangle.x1) / 2;
        var yCenter = rectangle.y1 + (rectangle.y2 - rectangle.y1) / 2;
        console.log(xCenter - LINE_OFFSET, xCenter + LINE_OFFSET);

        if (rectangle.x1 - LINE_OFFSET < x && x < rectangle.x1 + LINE_OFFSET) {
          if (
            rectangle.y1 - LINE_OFFSET < y &&
            y < rectangle.y1 + LINE_OFFSET
          ) {
            return { index: i, pos: "tl" };
          } else if (
            rectangle.y2 - LINE_OFFSET < y &&
            y < rectangle.y2 + LINE_OFFSET
          ) {
            return { index: i, pos: "bl" };
          } else if (yCenter - LINE_OFFSET < y && y < yCenter + LINE_OFFSET) {
            return { index: i, pos: "l" };
          }
        } else if (
          rectangle.x2 - LINE_OFFSET < x &&
          x < rectangle.x2 + LINE_OFFSET
        ) {
          if (
            rectangle.y1 - LINE_OFFSET < y &&
            y < rectangle.y1 + LINE_OFFSET
          ) {
            return { index: i, pos: "tr" };
          } else if (
            rectangle.y2 - LINE_OFFSET < y &&
            y < rectangle.y2 + LINE_OFFSET
          ) {
            return { index: i, pos: "br" };
          } else if (yCenter - LINE_OFFSET < y && y < yCenter + LINE_OFFSET) {
            return { index: i, pos: "r" };
          }
        } else if (xCenter - LINE_OFFSET < x && x < xCenter + LINE_OFFSET) {
          if (
            rectangle.y1 - LINE_OFFSET < y &&
            y < rectangle.y1 + LINE_OFFSET
          ) {
            return { index: i, pos: "t" };
          } else if (
            rectangle.y2 - LINE_OFFSET < y &&
            y < rectangle.y2 + LINE_OFFSET
          ) {
            return { index: i, pos: "b" };
          } else if (
            rectangle.y1 - LINE_OFFSET < y &&
            y < rectangle.y2 + LINE_OFFSET
          ) {
            return { index: i, pos: "i" };
          }
        } else if (
          rectangle.x1 - LINE_OFFSET < x &&
          x < rectangle.x2 + LINE_OFFSET
        ) {
          if (
            rectangle.y1 - LINE_OFFSET < y &&
            y < rectangle.y2 + LINE_OFFSET
          ) {
            return { index: i, pos: "i" };
          }
        }
      }
      return { index: -1, pos: "o" };
    }

    useEffect(() => {
      const canvas = canvasRef.current;
      const context = canvas.getContext("2d");
      const image = new Image();

      image.src = image_path;
      image.onload = () => {
        context.drawImage(image, 0, 0, canvas.width, canvas.height);
        rectangles.forEach(({ x1, y1, x2, y2, color }) => {
          drawRectangle(context, x1, y1, x2, y2, color);
        });

        if (clickedArea.index === -1 && currentRect) {
          drawRectangle(
            context,
            currentRect.x1,
            currentRect.y1,
            currentRect.x2,
            currentRect.y2,
            currentRect.label
          );
        }
      };

      const counts = rectangles.reduce((acc, rect) => {
        const label = rect.label || "Unlabeled";
        acc[label] = (acc[label] || 0) + 1;
        return acc;
      }, {});
      setLabelCounts(counts);
    }, [rectangles, currentRect]);

    const drawRectangle = (context, x1, y1, x2, y2, color, label = "") => {
      let width = x2 - x1;
      let height = y2 - y1;

      context.strokeStyle = color;
      context.lineWidth = 2;
      context.strokeRect(x1, y1, width, height);

      let xCenter = x1 + (x2 - x1) / 2;
      let yCenter = y1 + (y2 - y1) / 2;

      context.fillRect(
        x1 - ANCHOR_SIZE,
        y1 - ANCHOR_SIZE,
        2 * ANCHOR_SIZE,
        2 * ANCHOR_SIZE
      );
      context.fillRect(
        x1 - ANCHOR_SIZE,
        yCenter - ANCHOR_SIZE,
        2 * ANCHOR_SIZE,
        2 * ANCHOR_SIZE
      );
      context.fillRect(
        x1 - ANCHOR_SIZE,
        y2 - ANCHOR_SIZE,
        2 * ANCHOR_SIZE,
        2 * ANCHOR_SIZE
      );
      context.fillRect(
        xCenter - ANCHOR_SIZE,
        y1 - ANCHOR_SIZE,
        2 * ANCHOR_SIZE,
        2 * ANCHOR_SIZE
      );
      context.fillRect(
        xCenter - ANCHOR_SIZE,
        yCenter - ANCHOR_SIZE,
        2 * ANCHOR_SIZE,
        2 * ANCHOR_SIZE
      );
      context.fillRect(
        xCenter - ANCHOR_SIZE,
        y2 - ANCHOR_SIZE,
        2 * ANCHOR_SIZE,
        2 * ANCHOR_SIZE
      );
      context.fillRect(
        x2 - ANCHOR_SIZE,
        y1 - ANCHOR_SIZE,
        2 * ANCHOR_SIZE,
        2 * ANCHOR_SIZE
      );
      context.fillRect(
        x2 - ANCHOR_SIZE,
        yCenter - ANCHOR_SIZE,
        2 * ANCHOR_SIZE,
        2 * ANCHOR_SIZE
      );
      context.fillRect(
        x2 - ANCHOR_SIZE,
        y2 - ANCHOR_SIZE,
        2 * ANCHOR_SIZE,
        2 * ANCHOR_SIZE
      );

      if (label) {
        context.fillStyle = "black";
        context.fillText(label, x1, y1 - 5);
      }
    };

    const handleMouseDown = (e) => {
      setMouseDown(true);

      // find out where the click happened
      const canvas = canvasRef.current;
      const rect = canvas.getBoundingClientRect();
      const x = e.clientX - rect.left;
      const y = e.clientY - rect.top;
      let currentArea = findCurrentArea(x, y);
      setClickedArea(currentArea);

      setCurrentRect({
        x1: x,
        y1: y,
        x2: x,
        y2: y,
      });

      if (currentArea.index === -1) {
        setLabel("");
      } else if (
        currentArea.index !== -1 &&
        rectangles[currentArea.index].label
      ) {
        setLabel(rectangles[currentArea.index].label);
      }
    };

    const handleMouseMove = (e) => {
      if (!mouseDown) {
        return;
      }

      const canvas = canvasRef.current;
      const rect = canvas.getBoundingClientRect();
      const x = e.clientX - rect.left;
      const y = e.clientY - rect.top;

      console.log("clickedArea", clickedArea);
      console.log("currentRect", currentRect);
      console.log(rectangles);

      // new rectangle
      if (clickedArea.index === -1) {
        setCurrentRect((prevRect) => ({
          ...prevRect,
          x2: x,
          y2: y,
        }));
        // adjust existing rectangle
      } else if (clickedArea.index !== -1) {
        let x1 = currentRect.x1;
        let y1 = currentRect.y1;
        let xOffset = x - x1;
        let yOffset = y - y1;

        setCurrentRect((prevRect) => ({
          ...prevRect,
          x1: x,
          y1: y,
        }));

        setRectangles((prevRectangles) => {
          const newRectangles = [...prevRectangles];
          const selectedBox = { ...newRectangles[clickedArea.index] };

          if (
            clickedArea.pos === "i" ||
            clickedArea.pos === "tl" ||
            clickedArea.pos === "l" ||
            clickedArea.pos === "bl"
          ) {
            selectedBox.x1 += xOffset;
          }
          if (
            clickedArea.pos === "i" ||
            clickedArea.pos === "tl" ||
            clickedArea.pos === "t" ||
            clickedArea.pos === "tr"
          ) {
            selectedBox.y1 += yOffset;
          }
          if (
            clickedArea.pos === "i" ||
            clickedArea.pos === "tr" ||
            clickedArea.pos === "r" ||
            clickedArea.pos === "br"
          ) {
            selectedBox.x2 += xOffset;
          }
          if (
            clickedArea.pos === "i" ||
            clickedArea.pos === "bl" ||
            clickedArea.pos === "b" ||
            clickedArea.pos === "br"
          ) {
            selectedBox.y2 += yOffset;
          }

          newRectangles[clickedArea.index] = selectedBox;
          return newRectangles;
        });
      }
    };

    const handleMouseUp = () => {
      setRectangles((prevRectangles) => {
        if (clickedArea.index === -1 && currentRect) {
          return [...prevRectangles, currentRect];
        } else if (clickedArea.index !== -1) {
          const newRectangles = [...prevRectangles];
          const selectedBox = { ...newRectangles[clickedArea.index] };

          if (selectedBox.x1 > selectedBox.x2) {
            selectedBox.x1, (selectedBox.x2 = selectedBox.x2), selectedBox.x1;
          }
          if (selectedBox.y1 > selectedBox.y2) {
            selectedBox.y1, (selectedBox.y2 = selectedBox.y2), selectedBox.y1;
          }

          newRectangles[clickedArea.index] = selectedBox;
          return newRectangles;
        }

        return prevRectangles;
      });

      setCurrentRect(null);
      setMouseDown(false);
    };

    const handleLabelChange = (e) => {
      setLabel(e.target.value);

      setRectangles((prevRectangles) => {
        const newRectangles = [...prevRectangles];
        let i = clickedArea.index;
        // newly added rectangle
        if (clickedArea.index === -1) {
          i = rectangles.length - 1;
        }
        const selectedBox = {
          ...newRectangles[i],
          label: e.target.value,
        };
        newRectangles[i] = selectedBox;
        return newRectangles;
      });

      console.log(rectangles);
    };

    return (
      <main>
        <div className="flex flex-row">
          <div className="bg-[#3B3840] drop-shadow absolute top-5 left-5 rounded-lg p-4">
            <input
              className="bg-[#3B3840] border-b border-b-[#3FEABF] mb-5"
              type="text"
              value={label}
              onChange={handleLabelChange}
              placeholder="Enter label"
            />
            {Object.entries(labelCounts).map(([label, count]) => (
              <div key={label} className="flex justify-between">
                <span>{label}</span>
                <span>{count}</span>
              </div>
            ))}
          </div>
          <div className="bg-white border-2 mt-32 ml-64 h-100% hidden lg:block"></div>
          <div className="relative">
            <canvas
              ref={canvasRef}
              width={1000}
              height={500}
              className="border border-black"
              onMouseDown={handleMouseDown}
              onMouseMove={handleMouseMove}
              onMouseUp={handleMouseUp}
              onMouseOut={handleMouseUp}
            />
          </div>
        </div>
      </main>
    );
  }
}
