"use client";

import { useState } from "react";
import { toast } from "react-hot-toast";

import Navbar from "@/components/nav/NavBar";
import SentimentAnalysisSideNav from "@/components/nav/SentimentAnalysisSideNav";
import axios from "axios";

const modelData = [
  {
    model_name: "Support Vector Machine (SVM)",
    model_description: "Classifies text into sentiment categories by finding the optimal separating hyperplane. Effective in high-dimensional spaces."
  },
  {
    model_name: "Naive Bayes",
    model_description: "Probabilistic model based on feature independence assumptions. Suitable for text data with high dimensionality."
  },
  {
    model_name: "Random Forest",
    model_description: "Ensemble method using multiple decision trees. Robust to overfitting and captures complex interactions between features."
  },
  {
    model_name: "XGBoost (Extreme Gradient Boosting)",
    model_description: "Advanced boosting algorithm that builds decision trees sequentially with regularization. High performance and scalability."
  }
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

        toast.success("Job created")
      try {
        const response = await axios.post(apiEndpoint, selectedModel);
        console.log(response)
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
              <p className="mb-2">Select Model</p>
              {modelData.length === 0 && <p>No models found</p>}
              <div className="flex flex-row flex-wrap gap-x-4 gap-y-1">
                {modelData.map((model, index) => (
                  <div
                    key={index}
                    className={`flex flex-wrap border border-white border-opacity-50 w-72 rounded-lg cursor-pointer mt-1 p-4 ${
                      selectedModel?.model_name === model.model_name
                        ? "bg-[#3FEABF] text-black"
                        : "hover:bg-[#3FEABF] hover:text-black"
                    }`}
                    onClick={() => {
                      setSelectedModel(model);
                    }}
                  >
                    <p className="font-bold mb-2">{model.model_name}</p>
                    <p>{model.model_description}</p>
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
