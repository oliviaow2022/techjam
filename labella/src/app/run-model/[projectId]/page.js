"use client";

import axios from "axios";
import { useState, useEffect } from "react";
import Navbar from "@/components/NavBar";
import SideNav from "@/components/SideNav";

export default function RunModel({ params }) {
  let getModelsEndpoint =
    process.env.NEXT_PUBLIC_API_ENDPOINT +
    `/project/${params.projectId}/models`;

  const [modelData, setModelData] = useState([]);
  const [selectedModelId, setSelectedModelId] = useState([]);

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
      let response = await axios.post(
        process.env.NEXT_PUBLIC_API_ENDPOINT + `/model/${modelId}/label`
      );
      if (response.status === 200) {
        toast.success("Job created");
      }
    } catch (error) {
      console.log(error);
    }
  };

  return (
    <main className="flex flex-col min-h-screen px-24 pb-24 bg-[#19151E] z-20">
      <Navbar />
      <div className="flex flex-row">
        <SideNav params={params.projectId} />
        <div className="ml-0 lg:ml-20 mt-32">
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
          <button
            type="submit"
            className="flex p-2 bg-[#FF52BF] w-32 rounded-lg justify-center items-center cursor-pointer text-white"
            onClick={handleSubmit}
          >
            Run Model
          </button>
        </div>
      </div>
    </main>
  );
}
