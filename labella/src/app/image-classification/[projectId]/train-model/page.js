"use client";

import { useState } from "react";
import InputBox from "@/components/forms/InputBox";
import ImageClassificationSideNav from "@/components/nav/ImageClassificationSideNav";
import Navbar from "@/components/nav/NavBar";
import axios from "axios";
import { toast } from 'react-hot-toast'

const models = ["resnet18", "densenet121", "alexnet", "convnext_base"];

export default function TrainModelButton({ params }) {
  const [formData, setFormData] = useState({
    batch_size: 128,
    num_epochs: 3,
    train_test_split: 0.8,
    model_name: "",
  });

  const [errors, setErrors] = useState({});

  const handleSubmit = async (e) => {
    e.preventDefault();

    let validationErrors = validate()
    setErrors(validationErrors);

    if (Object.keys(validationErrors).length === 0) {
      try {
        let apiEndpoint =
          process.env.NEXT_PUBLIC_API_ENDPOINT + `/model/${params.projectId}/train`;
          console.log(formData)
        const response = await axios.post(apiEndpoint, formData);
        
        console.log(response)

        if (response.status === 200) {
          toast.success("Job created");
        }
      } catch (err) {
        console.log(err);
      } 
    }
  };

  const handleChange = (e) => {
    const { name, value } = e.target;

    setFormData({
      ...formData,
      [name]: value,
    });
  };

  const validate = () => {
    let errors = {};

    if (!formData.batch_size) {
      errors.batch_size = "Batch size is required";
    }
    if (!formData.num_epochs) {
      errors.num_epochs = "Number of epochs is required";
    }
    if (!formData.train_test_split) {
      errors.train_test_split = "Train test split ratio is required";
    }
    if (!formData.model_name) {
      errors.model_name = "Model is required";
    }

    return errors;
  };

  return (
    <main className="flex flex-col min-h-screen px-24 pb-24 bg-[#19151E] z-20">
      <Navbar />
      <div className="flex flex-row">
        <ImageClassificationSideNav params={params.projectId} />
        <div className="ml-0 lg:ml-20">
          <p
            className="text-xl text-[#FF52BF] font-bold mb-8 mt-40"
            id="Train Model"
          >
            Train Model
          </p>
          <form>
            <p className="mt-2">Select Model Architecture</p>
            <div className="mb-4 grid grid-cols-2 sm:grid-cols-4 gap-6">
              {models.map((model, index) => (
                <div
                  key={index}
                  className={`flex border border-white border-opacity-50 w-32 xl:w-40 2xl:w-64 items-center justify-center rounded-lg h-8 cursor-pointer my-1 ${
                    formData.model_name === model
                      ? "bg-[#FF52BF] text-black"
                      : "hover:bg-[#FF52BF] hover:text-black"
                  }`}
                  onClick={() => {
                    setFormData({ ...formData, model_name: model });
                  }}
                >
                  {model}
                </div>
              ))}
              {errors.model_name && (
                <p className="text-red-500 text-sm">{errors.model_name}</p>
              )}
            </div>
            <div className="mb-4">
              <InputBox
                label={"Batch size"}
                name="batch_size"
                value={formData.batch_size}
                onChange={handleChange}
                error={errors.batch_size}
              />
            </div>
            <div className="mb-4">
              <InputBox
                label={"Number of training epochs"}
                name="num_epochs"
                value={formData.num_epochs}
                onChange={handleChange}
                error={errors.num_epochs}
              />
            </div>
            <div className="mb-4">
              <InputBox
                label={"Train-test split ratio"}
                name="train_test_split"
                value={formData.train_test_split}
                onChange={handleChange}
                error={errors.train_test_split}
              />
            </div>

            <button
              type="submit"
              className="flex p-2 bg-[#FF52BF] w-32 rounded-lg justify-center items-center cursor-pointer text-white"
              onClick={handleSubmit}
            >
              Train Model
            </button>
          </form>
        </div>
      </div>
    </main>
  );
}