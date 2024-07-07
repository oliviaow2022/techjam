"use client";

import axios from "axios";
import { useState, useEffect } from "react";
import Navbar from "@/components/nav/NavBar";
import ImageClassificationSideNav from "@/components/nav/ImageClassificationSideNav";
import { toast } from "react-hot-toast";

export default function DownloadModel({ params }) {
  let getModelsEndpoint =
    process.env.NEXT_PUBLIC_API_ENDPOINT +
    `/project/${params.projectId}/models`;

  const [modelData, setModelData] = useState([]);
  const [error, setError] = useState('');
  const [selectedModelId, setSelectedModelId] = useState(null);

  useEffect(() => {
    const fetchModelData = async () => {
      try {
        let response = await axios.get(getModelsEndpoint);
        console.log(response.data);
        setModelData(response.data);
      } catch (error) {
        console.log(error);
      }
    };

    fetchModelData();
  }, []);

  const handleSubmit = async () => {
    try {
      if (!selectedModelId) {
        setError('Please select a model')
        toast.error('Model missing')
        return
      }

      toast.success("Downloading file");
      let response = await axios.get(
        process.env.NEXT_PUBLIC_API_ENDPOINT +
          `/model/${selectedModelId}/download`,
        {
          responseType: "blob", // Important: 'blob' indicates binary data
        }
      );
      console.log(response)
      if (response.status === 200) {
        const url = window.URL.createObjectURL(new Blob([response.data]));
        const link = document.createElement('a');
        link.href = url;
        link.setAttribute('download', `model.pth`); // Set the file name here
        document.body.appendChild(link);
        link.click();
        document.body.removeChild(link);
      }
    } catch (error) {
      console.log(error);
    }
  };

  return (
    <main className="flex flex-col min-h-screen px-24 pb-24 bg-[#19151E] z-20">
      <Navbar />
      <div className="flex flex-row">
        <ImageClassificationSideNav params={params.projectId} />
        <div className="ml-0 lg:ml-20 mt-32">
          <div>
          <p className="font-bold mb-2">Select Trained Model</p>
          {modelData.length === 0 && <p>No trained models found</p>}
          <div className="mb-4 grid grid-cols-2 sm:grid-cols-4 gap-6">
            {modelData.map((model, index) => (
              <div
                key={index}
                className={`flex border border-white border-opacity-50 w-32 xl:w-40 2xl:w-64 items-center justify-center rounded-lg h-8 cursor-pointer my-1 ${
                  selectedModelId === model.id
                    ? "bg-[#FF52BF] text-black"
                    : "hover:bg-[#FF52BF] hover:text-black"
                }`}
                onClick={() => {
                  setSelectedModelId(model.id);
                }}
              >
                {model.name}
              </div>
            ))}
          </div>
          {error && <p className="text-red-500 text-sm mb-4">{error}</p>}
          </div>

          <button
            type="submit"
            className="flex py-2 px-4 bg-[#FF52BF] w-fit rounded-lg justify-center items-center cursor-pointer text-white"
            onClick={handleSubmit}
          >
            Download Model
          </button>
        </div>
      </div>
    </main>
  );
}
