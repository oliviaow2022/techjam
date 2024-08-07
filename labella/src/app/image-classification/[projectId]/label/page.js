"use client";
import React, { useState, useEffect } from "react";
import LabelButton from "@/components/LabelButton";
import ImageSlider from "@/components/ImageSlider";
import ImageClassificationSideNav from "@/components/nav/ImageClassificationSideNav";
import axios from "axios";
import Navbar from "@/components/nav/NavBar";
import { toast } from "react-hot-toast";

export default function Label({ params }) {
  const batchApiEndpoint =
    process.env.NEXT_PUBLIC_API_ENDPOINT + `/dataset/${params.projectId}/batch`;
  const datasetApiEndpoint =
    process.env.NEXT_PUBLIC_API_ENDPOINT + `/dataset/${params.projectId}`;
  const jwtToken = localStorage.getItem("jwt");

  const config = {
    headers: {
      "Content-Type": "application/json",
      Authorization: `Bearer ${jwtToken}`,
    },
  };

  const [images, setImages] = useState([]);
  const [datasetData, setDatasetData] = useState({});
  const [error, setError] = useState(null);
  const [loading, setLoading] = useState(true);
  const [currentIndex, setCurrentIndex] = useState(0);
  const [multilabelslist, setMultilabelslist] = useState([]);
  const fetchBatch = async () => {
    try {
      const batchResponse = await axios.post(batchApiEndpoint, {});
      setImages(batchResponse.data);
      console.log(batchResponse.data);
    } catch (error) {
      setError(error.message);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    // to get class_to_label_mapping
    const fetchDataset = async () => {
      try {
        const datasetResponse = await axios.get(datasetApiEndpoint);
        setDatasetData(datasetResponse.data);
      } catch (error) {
        setError(error.message);
      } finally {
        setLoading(false);
      }
    };

    fetchDataset();
    fetchBatch();
  }, [batchApiEndpoint, datasetApiEndpoint]);

  useEffect(() => {
    if (currentIndex === images.length) {
      fetchBatch();
      setCurrentIndex(0);
    }
  }, [currentIndex, images.length]);

  if (loading) {
    return <div>Loading...</div>;
  }
  if (error) {
    return <div>Error: {error}</div>;
  }

  const handleImageChange = (idx) => {
    setCurrentIndex(idx);
  };

  const handleLabelAdditionSingle = async (classInteger) => {
    let apiEndpoint =
      process.env.NEXT_PUBLIC_API_ENDPOINT +
      `/instance/${images[currentIndex].id}/set_label`;
    try {
      const response = await axios.post(apiEndpoint, {
        labels: classInteger,
      });
      if (response.status === 200) {
        setCurrentIndex((currentIndex) => (currentIndex += 1));
        toast.success("Label updated");
        const updatedImages = images.map((img, idx) =>
          idx === currentIndex ? { ...img, labels: classInteger } : img
        );
        setImages(updatedImages);
      }
    } catch (err) {
      console.log(err);
    }
  };
  
  const handleLabelAdditionMulti = async (classInteger) => {
    const list = classInteger.split("_")
    const categoryIndex = list[0]
    const categoryInteger = list[1]
    const updatedLabelsList = [...multilabelslist]
    updatedLabelsList[categoryIndex] = categoryInteger
    setMultilabelslist(updatedLabelsList)
    const formattedClassInteger = updatedLabelsList.join(',');
    
    if (/^\d+(,\d+)*$/.test(formattedClassInteger)) {
      // Send updated labels to the backend
      let apiEndpoint =
        process.env.NEXT_PUBLIC_API_ENDPOINT +
        `/instance/${images[currentIndex].id}/set_label`;
      try {
        const response = await axios.post(apiEndpoint, {
          labels: formattedClassInteger,
        });
        if (response.status === 200) {
          toast.success("Labels updated");
          const updatedImages = images.map((img, idx) =>
            idx === currentIndex ? { ...img, labels: updatedLabelsList } : img
          );
          setImages(updatedImages);
        }
      } catch (err) {
        console.error('Error updating labels:', err);
      }
    }
  }

  const parseJsonIfNeeded = (data) => {
    if (typeof data === 'string') {
      try {
        return JSON.parse(data);
      } catch (error) {
        console.error('Failed to parse JSON string:', error);
        return {};
      }
    }
    return data;
  };
  const parsedClassToLabelMapping = parseJsonIfNeeded(datasetData?.dataset?.class_to_label_mapping)  
  return (
    <main className="flex flex-col min-h-screen px-24 pb-24 bg-[#19151E] z-20">
      <Navbar />
      <div className="flex flex-row">
        <ImageClassificationSideNav params={params.projectId} />
        <div className="ml-0 lg:ml-20">
          <p className="text-xl text-[#FF52BF] font-bold mb-8 mt-40">
            Image Classification
          </p>
          <div className="flex gap-4 mb-4 items-center justify-center">
            {images && (
              <ImageSlider
                images={images}
                bucketname={datasetData.project.bucket}
                bucketprefix={datasetData.project.prefix}
                handleImageChange={handleImageChange}
                currentIndex={currentIndex}
                setCurrentIndex={setCurrentIndex}
              />
            )}
            <div className="flex flex-col gap-y-4">
              {datasetData?.dataset.project_type === "Multilabel Classification" ?  (<div className="bg-[#3B3840] rounded-lg w-96 p-4">
                  {Object.entries(parsedClassToLabelMapping).map(([category, label], categoryIndex) => {
                    return (
                      <div>
                        <p className="text-white font-bold mb-2">{category}</p>
                        <div className="flex flex-wrap gap-4">
                          {Object.entries(label).map(([key, value]) => (
                            <LabelButton
                              key={key}
                              classInteger={`${categoryIndex}_${key}`}
                              name={value}
                              handleOptionChange={handleLabelAdditionMulti}
                              selectedOption={images[currentIndex]}
                              isMulticlass={true}
                            />
                          ))}
                        </div>
                      </div>)
                  })}
                </div>) : (<div className="bg-[#3B3840] rounded-lg w-96 p-4">
                <p className="text-white font-bold mb-2">Class</p>
                <div className="flex flex-wrap gap-4">
                  {Object.entries(parsedClassToLabelMapping).map(
                    ([key, value]) => (
                      <LabelButton
                        key={key}
                        classInteger={key}
                        name={value}
                        handleOptionChange={handleLabelAdditionSingle}
                        selectedOption={images[currentIndex]}
                        isMulticlass={false}
                      />
                    )
                  )}
                </div>
              </div>)}
              <div className="bg-[#3B3840] rounded-lg w-96 p-4">
                <p className="text-white font-bold mb-2">Entropy</p>
                <p>{images[currentIndex]?.entropy}</p>
              </div>
            </div>
          </div>
          <div className="grid grid-cols-10 gap-2">
            {images.map((item, index) => {
              return (
                <div key={item.data} onClick={() => setCurrentIndex(index)}>
                  <img
                    src={`https://${datasetData?.project.bucket}.s3.amazonaws.com/${datasetData?.project.prefix}/${item.data}`}
                    className={`w-20 h-20 rounded-lg ${
                      currentIndex === index ? "border-4 border-white" : ""
                    }`}
                  />
                  <p className="break-all" style={{ fontSize: "10px" }}>
                    {item.data}
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
