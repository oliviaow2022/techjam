"use client";

import { useState } from "react";
import { toast } from "react-hot-toast";

import Navbar from "@/components/nav/NavBar";
import SentimentAnalysisSideNav from "@/components/nav/SentimentAnalysisSideNav";
import axios from "axios";

const modelData = [
  {
    name: "Support Vector Classifier",
    value: "SVC",
  },
];

export default function TrainModel({ params }) {
  const [errors, setErrors] = useState({});
  const [selectedModel, setSelectedModel] = useState(null);

  const handleSubmit = async (e) => {
    e.preventDefault();
    let validationErrors = validate();
    setErrors(validationErrors);

    if (Object.keys(validationErrors).length === 0) {
      let apiEndpoint =
        process.env.NEXT_PUBLIC_API_ENDPOINT + `/senti/${params.projectId}/train`;

      try {
        const response = await axios.post(apiEndpoint, {
          model_name: selectedModel,
        });
        if (response.status === 200) {
          toast.success("Job created")
        }
      } catch (error) {
        toast.error("Error");
        console.log(error);
      }
    }
  };

  const validate = () => {
    let errors = {};

    if (!selectedModel) {
      errors.selectedModel = "Please select a model";
    }
    return errors;
  };

  return (
    <main className="flex flex-col min-h-screen px-24 pb-24 bg-[#19151E] z-20">
      <Navbar />
      <div className="flex flex-row">
        <SentimentAnalysisSideNav params={params.projectId} />
        <form className="ml-0 lg:ml-20 mt-32" onSubmit={handleSubmit}>
          <p className="text-xl text-[#3FEABF] font-bold mb-8">
            Sentiment Analysis
          </p>
          <div className="flex flex-col gap-y-5">
            <div>
              <p className="mb-2">Select Zero-Shot Model</p>
              {modelData.length === 0 && <p>No models found</p>}
              <div className="grid lg:grid-cols-2 gap-x-6 gap-y-1">
                {modelData.map((model, index) => (
                  <div
                    key={index}
                    className={`flex flex-wrap border border-white border-opacity-50 w-72 items-center justify-center rounded-lg h-8 cursor-pointer mt-1 ${
                      selectedModel === model.value
                        ? "bg-[#3FEABF] text-black"
                        : "hover:bg-[#3FEABF] hover:text-black"
                    }`}
                    onClick={() => {
                      setSelectedModel(model.value);
                    }}
                  >
                    {model.name}
                  </div>
                ))}
              </div>
              {errors.selectedModel && (
                <p className="text-red-500 text-sm">{errors.selectedModel}</p>
              )}
            </div>
            <button
              type="submit"
              className="flex my-4 py-2 px-4 bg-[#FF52BF] w-fit rounded-lg justify-center items-center cursor-pointer text-white"
            >
              Train Model
            </button>
          </div>
        </form>
      </div>
    </main>
  );
}
