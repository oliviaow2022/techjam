"use client";

import { toast } from "react-hot-toast";
import Arrow from "@/components/Arrow";
import { useState, useEffect } from "react";

import Navbar from "@/components/nav/NavBar";
import SentimentAnalysisSideNav from "@/components/nav/SentimentAnalysisSideNav";
import LabelButton from "@/components/LabelButton";
import axios from "axios";

export default function SentimentAnalysisLabelling({ params }) {
  const datasetApiEndpoint =
    process.env.NEXT_PUBLIC_API_ENDPOINT + `/dataset/${params.projectId}`;
  const batchApiEndpoint =
    process.env.NEXT_PUBLIC_API_ENDPOINT + `/dataset/${params.projectId}/batch`;

  const [loading, setLoading] = useState(true);
  const [datasetData, setDatasetData] = useState({});
  const [batchData, setBatchData] = useState([]);
  const [error, setError] = useState(null);
  const [currentIndex, setCurrentIndex] = useState(0);

  const handleLabelAddition = async (classInteger) => {
    let apiEndpoint =
      process.env.NEXT_PUBLIC_API_ENDPOINT +
      `/instance/${batchData[currentIndex].id}/set_label`;

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

  useEffect(() => {
    const fetchDataset = async () => {
      try {
        const datasetResponse = await axios.get(datasetApiEndpoint);
        console.log(datasetResponse.data);
        setDatasetData(datasetResponse.data);
      } catch (error) {
        setError(error.message);
      } finally {
        setLoading(false);
      }
    };

    const fetchBatch = async () => {
      try {
        const batchResponse = await axios.post(batchApiEndpoint, {
          batch_size: 20,
        });
        setBatchData(batchResponse.data);
        console.log(batchResponse.data);
      } catch (error) {
        setError(error.message);
      } finally {
        setLoading(false);
      }
    };

    fetchDataset();
    fetchBatch();
  }, [datasetApiEndpoint, batchApiEndpoint]);

  if (loading) {
    return <div>Loading...</div>;
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
        <SentimentAnalysisSideNav params={params.projectId} />
        <div className="ml-0 lg:ml-20 mt-32">
          <p className="text-xl text-[#3FEABF] font-bold mb-8">
            Sentiment Analysis
          </p>
          <div className="flex flex-row gap-4">
            <div className="flex gap-4 items-center justify-center">
              <button
                onClick={() => setCurrentIndex((prevIndex) => prevIndex - 1)}
              >
                <Arrow direction="left" />
              </button>
              <div className="w-96 h-64 rounded-lg border-2 border-white p-4 overflow-y-scroll">
                <p className="text-white font-bold mb-2">Data</p>
                {batchData[currentIndex]?.data}
              </div>
              <button
                onClick={() => setCurrentIndex((prevIndex) => prevIndex + 1)}
              >
                <Arrow direction="right" />
              </button>
            </div>

            <div className="flex flex-col gap-y-4">
              <div className="bg-[#3B3840] rounded-lg w-96 p-4">
                <p className="text-white font-bold mb-2">Class</p>
                <div className="flex flex-wrap justify-between">
                  {Object.entries(
                    parsedClassToLabelMapping
                  ).map(([key, value]) => (
                    <LabelButton
                      key={key}
                      classInteger={key}
                      name={value}
                      handleOptionChange={handleLabelAddition}
                      bgColour="bg-[#3FEABF]"
                    />
                  ))}
                </div>
              </div>
              <div className="bg-[#3B3840] rounded-lg w-96 p-4">
                <p className="text-white font-bold mb-2">Entropy</p>
                <p>{batchData[currentIndex]?.entropy}</p>
              </div>
            </div>
          </div>
          <div className="grid grid-cols-10 gap-2">
            {batchData.map((data_instance, index) => {
              return (
                <div
                  key={data_instance.id}
                  className={`w-20 h-20 rounded-lg overflow-hidden text-xs p-1 ${
                    currentIndex === index ? "border-4 border-white" : ""
                  }`}
                  onClick={() => setCurrentIndex(index)}
                >
                  {data_instance.data}
                </div>
              );
            })}
          </div>
        </div>
      </div>
    </main>
  );
}
