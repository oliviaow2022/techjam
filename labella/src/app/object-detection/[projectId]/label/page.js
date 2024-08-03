"use client";

import Navbar from "@/components/nav/NavBar";
import axios from "axios";
import ObjectDetectionSideNav from "@/components/nav/ObjectDetectionSideNav";
import Arrow from "@/components/Arrow";

import { RiDeleteBin5Fill } from "react-icons/ri";
import { useState, useEffect, useRef } from "react";
import { toast } from "react-hot-toast";

const MAX_WIDTH = 800;
const MAX_HEIGHT = 500;
const LINE_OFFSET = 6;
const ANCHOR_SIZE = 2;

export default function ObjectDetection({ params }) {
  {
    // fetch data logic
    const batchApiEndpoint =
      process.env.NEXT_PUBLIC_API_ENDPOINT +
      `/objdet/${params.projectId}/batch`;
    const datasetApiEndpoint =
      process.env.NEXT_PUBLIC_API_ENDPOINT + `/dataset/${params.projectId}`;

    const [isLoading, setIsLoading] = useState(false);
    const [images, setImages] = useState([]);
    const [currentIndex, setCurrentIndex] = useState(0);
    const [datasetData, setDatasetData] = useState({});
    const [imageDisplayRatio, setImageDisplayRatio] = useState(1);

    const fetchBatch = async () => {
      try {
        setIsLoading(true);
        const batchResponse = await axios.post(batchApiEndpoint, {});
        console.log(batchResponse.data);
        setImages(batchResponse.data);

        if (batchResponse.data[currentIndex].bboxes) {
          // scale coordinates from original image to canvas
          const scaledBboxes = batchResponse.data[currentIndex].bboxes.map(
            (rect) => {
              return {
                x1: Math.round(rect.x1 * imageDisplayRatio),
                y1: Math.round(rect.y1 * imageDisplayRatio),
                x2: Math.round(rect.x2 * imageDisplayRatio),
                y2: Math.round(rect.y2 * imageDisplayRatio),
                label: rect.label,
              };
            }
          );
          setRectangles(scaledBboxes);
        }
      } catch (error) {
        console.log(error);
      } finally {
        setIsLoading(false);
      }
    };

    useEffect(() => {
      // to get class_to_label_mapping
      const fetchDataset = async () => {
        try {
          setIsLoading(true);
          const datasetResponse = await axios.get(datasetApiEndpoint);
          setDatasetData(datasetResponse.data);
          console.log(datasetResponse.data);
        } catch (error) {
          console.log(error);
        } finally {
          setIsLoading(false);
        }
      };

      fetchDataset();
      fetchBatch();
    }, [batchApiEndpoint, datasetApiEndpoint]);

    useEffect(() => {
      if (currentIndex === images.length) {
        fetchBatch();
        setCurrentIndex(0);
        console.log("currentImage", currentIndex, images[currentIndex]);
      }
    }, [batchApiEndpoint, currentIndex, images.length]);

    // handle arrow click logic
    const updateLabel = async (currentIndex) => {
      if (rectangles.length == 0) {
        return;
      }

      rectangles.forEach((rect) => {
        if (!rect.label) {
          toast.error("Label missing");
          throw new Error("Label missing");
        }
      });

      console.log("updated currentImage", currentIndex, images[currentIndex]);

      let updateEndpoint =
        process.env.NEXT_PUBLIC_API_ENDPOINT +
        `/objdet/${images[currentIndex].id}/label`;

      const updateResponse = await axios.post(updateEndpoint, {
        annotations: rectangles,
        image_display_ratio: imageDisplayRatio,
      });

      if (updateResponse.status === 200) {
        toast.success("Label updated");
      }

      // update image bbox locally
      const updatedImages = [...images];
      const currentImage = updatedImages[currentIndex];
      currentImage.bboxes = rectangles;
      setImages(updatedImages);

      console.log(updateResponse.data);
    };

    const handlePrev = async () => {
      try {
        await updateLabel(currentIndex);
        const newIndex = (currentIndex - 1 + images.length) % images.length;
        setCurrentIndex(newIndex);
        console.log("currentImage", newIndex, images[newIndex]);
        resetCanvas(newIndex);
      } catch (error) {
        console.log(error);
      }
    };

    const handleNext = async () => {
      try {
        await updateLabel(currentIndex);
        const newIndex = currentIndex + 1;
        setCurrentIndex(newIndex);
        console.log("currentImage", newIndex, images[newIndex]);
        resetCanvas(newIndex);
      } catch (error) {
        console.log(error);
      }
    };

    const handleClick = async (index) => {
      try {
        await updateLabel(currentIndex);
        setCurrentIndex(index);
        console.log("currentImage", index, images[index]);
        resetCanvas(index);
      } catch (error) {
        console.log(error);
      }
    };

    // canvas logic
    const resetCanvas = (newIndex) => {
      // take existing bboxes
      if (images[newIndex].bboxes) {
        console.log(
          "loading bbox from current image",
          newIndex,
          images[newIndex]
        );

        // scale coordinates from original image to canvas
        const scaledBboxes = images[newIndex].bboxes.map((rect) => {
          return {
            x1: Math.round(rect.x1 * imageDisplayRatio),
            y1: Math.round(rect.y1 * imageDisplayRatio),
            x2: Math.round(rect.x2 * imageDisplayRatio),
            y2: Math.round(rect.y2 * imageDisplayRatio),
            label: rect.label,
          };
        });
        console.log("imageDisplayRatio", imageDisplayRatio);
        console.log("scaledBboxes", scaledBboxes);
        setRectangles(scaledBboxes);
      } else {
        setRectangles([]);
      }

      setLabelCounts({});
      setMouseDown(false);
      setCurrentRect(null);
      setClickedArea({ index: -1, pos: "o" });
      setLabel("");
    };

    const canvasRef = useRef(null);
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

    // redraw canvas when rectangles change
    useEffect(() => {
      const canvas = canvasRef.current;
      const context = canvas.getContext("2d");

      const image = new Image();
      if (datasetData && datasetData.project) {
        image.src = `https://${datasetData?.project.bucket}.s3.amazonaws.com/${datasetData?.project.prefix}/${images[currentIndex]?.filename}`;

        const aspectRatio = image.width / image.height;
        // Determine the canvas dimensions while maintaining the aspect ratio
        let canvasWidth = image.width;
        let canvasHeight = image.height;

        // use aspect ratio equation
        if (canvasWidth > MAX_WIDTH) {
          canvasWidth = MAX_WIDTH;
          canvasHeight = canvasWidth / aspectRatio;
        }

        if (canvasHeight > MAX_HEIGHT) {
          canvasHeight = MAX_HEIGHT;
          canvasWidth = canvasHeight * aspectRatio;
        }

        setImageDisplayRatio(image.width / canvasWidth);

        // Set the canvas dimensions
        canvas.width = canvasWidth;
        canvas.height = canvasHeight;

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
      }
    }, [rectangles, currentRect, currentIndex, images]);

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

      if (currentArea.index !== -1 && rectangles[currentArea.index].label) {
        setLabel(rectangles[currentArea.index].label);
      } else {
        setLabel("");
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
      console.log("rectangles", rectangles);

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

    function calculateRectangleArea(rectangle) {
      let width = Math.abs(rectangle.x2 - rectangle.x1);
      let height = Math.abs(rectangle.y2 - rectangle.y1);
      return width * height;
    }

    const handleMouseUp = () => {
      setRectangles((prevRectangles) => {
        // new rectangle
        if (clickedArea.index === -1 && currentRect) {
          // rectangle area too small
          if (calculateRectangleArea(currentRect) < 150) {
            return [...prevRectangles];
          }
          return [...prevRectangles, currentRect];
          // adjusting current rectangle
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

    // delete rectangle
    const deleteRectangleByIndex = (index) => {
      if (index < 0 || index >= rectangles.length) {
        toast.error("Invalid selection");
        return;
      }

      setRectangles((prevRectangles) => {
        const updatedRectangles = [...prevRectangles];
        updatedRectangles.splice(index, 1);
        return updatedRectangles;
      });

      setLabel("");
      setMouseDown(false);
      setCurrentRect(null);
      setClickedArea({ index: -1, pos: "o" });

      updateLabel(currentIndex);
    };

    return (
      <main className="flex flex-col min-h-screen px-24 pb-24 bg-[#19151E] z-20">
        <Navbar />
        <div className="flex flex-row">
          <ObjectDetectionSideNav params={params.projectId} />
          <div className="ml-0 mt-32">
            <p className="text-xl text-[#D887F5] font-bold mx-4 mb-4">
              Object Detection
            </p>
            <div className="flex flex-row">
              <div className="p-4">
                <div className="flex flex-row items-center mb-5">
                  <input
                    className="bg-transparent"
                    type="text"
                    value={label}
                    onChange={handleLabelChange}
                    placeholder="Enter label"
                  />
                  <button
                    className="rounded-lg"
                    onClick={() => deleteRectangleByIndex(clickedArea.index)}
                  >
                    <RiDeleteBin5Fill color="#D887F5" />
                  </button>
                </div>
                {Object.entries(labelCounts).map(([label, count]) => (
                  <div key={label} className="flex justify-between">
                    <span>{label}</span>
                    <span className="w-4 text-center">{count}</span>
                  </div>
                ))}
              </div>
              <div className="flex justify-between w-full">
                <button onClick={handlePrev}>
                  <Arrow direction="left" />
                </button>
                <canvas
                  height={500}
                  width={800}
                  ref={canvasRef}
                  className="border border-black"
                  onMouseDown={handleMouseDown}
                  onMouseMove={handleMouseMove}
                  onMouseUp={handleMouseUp}
                  onMouseOut={handleMouseUp}
                />
                <button onClick={handleNext}>
                  <Arrow direction="right" />
                </button>
              </div>
            </div>
            <div className="grid grid-cols-10 gap-2 p-4">
              {images.map((item, index) => {
                return (
                  <div
                    className="flex flex-col justify-center"
                    key={item.data}
                    onClick={() => handleClick(index)}
                  >
                    <img
                      src={`https://${datasetData?.project.bucket}.s3.amazonaws.com/${datasetData?.project.prefix}/${item.filename}`}
                      className={`w-20 h-20 rounded-lg ${
                        currentIndex === index
                          ? "border-4 border-[#D887F5]"
                          : ""
                      }`}
                    />
                    <p className="break-all" style={{ fontSize: "10px" }}>
                      {item.filename}
                    </p>
                  </div>
                );
              })}
            </div>
          </div>
        </div>
      </main>
    );
  }
}
