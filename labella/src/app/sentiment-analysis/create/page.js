"use client";

import { useState } from "react";
import Navbar from "@/components/nav/NavBar";
import ImageClassificationSideNav from "@/components/nav/ImageClassificationSideNav";
import FileInput from "@/components/forms/FileInput";
import CategoryInput from "@/components/forms/CategoryInput";
import InputBox from "@/components/forms/InputBox";

const modelData = [
  "DeBERTa-v3-base-mnli-fever-anli",
  "facebook/bart-large-mnli",
];

export default function SentimentAnalysis({ params }) {
  const [formData, setFormData] = useState({
    selectedModel: null,
    maxIterations: null,
    numSamples: null,
  });

  const [errors, setErrors] = useState({});
  const [zipFile, setZipFile] = useState(null);
  const [categoryList, setCategoryList] = useState([]);
  const [preprocessingSteps, setPreprocessingSteps] = useState([]);

  const handleFileChange = (e) => {
    console.log(e.target.files[0]);
    setZipFile(e.target.files[0]);
  };

  const handleChange = (e) => {
    const { name, value } = e.target;

    setFormData({
      ...formData,
      [name]: value,
    });
  };

  const handleSubmit = async () => {
    e.preventDefault()
    let validationErrors = validate()
    setErrors(validationErrors)
  }

  const validate = () => {
    let errors = {};

    if (!formData.selectedModel) {
      errors.selectedModel = "Please select a model";
    }
    if (!categoryList) {
      errors.categoryList = "Missing sentiment categories";
    }
    if (!preprocessingSteps) {
      errors.preprocessingSteps = "Missing preprocessing steps";
    }
    if (!formData.maxIterations) {
      errors.maxIterations = "Number of iterations is required";
    }
    if (!formData.numSamples) {
      errors.numSamples = "Number of samples is required";
    }
    if (!zipFile) {
      errors.zipFile = "Please select a file to upload.";
    }
    return errors;
  };

  return (
    <main className="flex flex-col min-h-screen px-24 pb-24 bg-[#19151E] z-20">
      <Navbar />
      <div className="flex flex-row">
        <ImageClassificationSideNav params={params.projectId} />
        <form className="ml-0 lg:ml-20 mt-32">
          <p className="text-xl text-[#3FEABF] font-bold mb-8">
            Sentiment Analysis
          </p>
          <div className="flex flex-col gap-y-5">
            <div>
              <p className="font-bold mb-2">Select Zero-Shot Model</p>
              {modelData.length === 0 && <p>No models found</p>}
              <div className="grid lg:grid-cols-2 gap-x-6 gap-y-1">
                {modelData.map((model, index) => (
                  <div
                    key={index}
                    className={`flex flex-wrap border border-white border-opacity-50 w-72 items-center justify-center rounded-lg h-8 cursor-pointer mt-1 ${
                      formData.selectedModel === model
                        ? "bg-[#3FEABF] text-black"
                        : "hover:bg-[#3FEABF] hover:text-black"
                    }`}
                    onClick={() => {
                      setFormData({
                        ...formData,
                        selectedModel: model,
                      });
                    }}
                  >
                    {model}
                  </div>
                ))}
              </div>
              {errors.selectedModel && (
                <p className="text-red-500 text-sm">{errors.selectedModel}</p>
              )}
            </div>
            <FileInput 
              label='Upload Dataset (CSV/XLSX only)'
              handleFileChange={handleFileChange}
              error={errors.zipFile}
            />
            <CategoryInput 
              label="Sentiment Categories"
              categoryList={categoryList}
              setCategoryList={setCategoryList}
              error={errors.categories}
            />
            <CategoryInput 
              label="Preprocessing Steps"
              categoryList={preprocessingSteps}
              setCategoryList={setPreprocessingSteps}
              error={errors.preprocessingSteps}
            />
            <InputBox 
              label="Maximum Iterations of Active Learning"
              name="maxIterations"
              value={formData.maxIterations}
              onChange={handleChange}
              error={errors.maxIterations}
            />
            <InputBox 
              label="Number of Samples for Manual Labelling"
              name="numSamples"
              value={formData.numSamples}
              onChange={handleChange}
              error={errors.numSamples}
            />
            <button
              type="submit"
              className="flex my-4 py-2 px-4 bg-[#FF52BF] w-fit rounded-lg justify-center items-center cursor-pointer text-white"
              onClick={handleSubmit}
            >
              Fine Tune Model
            </button>
          </div>
        </form>
      </div>
    </main>
  );
}
