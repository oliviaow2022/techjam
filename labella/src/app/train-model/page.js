"use client";

import { useState } from "react";
import InputBox from "@/components/InputBox";
import SideNav from "@/components/SideNav";
import Navbar from "@/components/NavBar";
import axios from "axios";

// user shld select model from db
const modelId = 4;

export default function TrainModelButton() {

  const [formData, setFormData] = useState({
    batch_size: 128,
    num_epochs: 3,
    train_test_split: 0.8,
  });
  const [errors, setErrors] = useState({});
  const [isSubmitting, setIsSubmitting] = useState(false);

  const handleSubmit = async (e) => {
    e.preventDefault();
    setErrors(validate());
    if (Object.keys(validationErrors).length === 0) {
      setIsSubmitting(true);
      try {
        let apiEndpoint =
          process.env.NEXT_PUBLIC_API_ENDPOINT + `/model/${modelId}/train`;
        const response = await axios.post(apiEndpoint, formData);
        console.log(response);
      } catch (err) {
        console.log(err);
      } finally {
        setIsSubmitting(false);
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

    if (!formData.batchSize) {
      errors.batchSize = "Batch size is required";
    }
    if (!formData.epochs) {
      errors.epochs = "Number of epochs is required";
    }
    if (!formData.splitRatio) {
      errors.splitRatio = "Train test split ratio is required";
    }
  };

  return (
    <main className="flex flex-col min-h-screen px-24 pb-24 bg-[#19151E] z-20">
      <Navbar />
      <div className="flex flex-row">
        <SideNav params={1} />
        <div className="bg-white border-2 mt-32 ml-64 h-100% hidden lg:block"></div>
        <div className="ml-0 lg:ml-20">
          <p
            className="text-xl text-[#FF52BF] font-bold mb-8 mt-40"
            id="Train Model"
          >
            Train Model
          </p>
          <form>
            <div className="mb-4">
              <InputBox
                label={"Batch size"}
                name="batchSize"
                value={formData.batchSize}
                onChange={handleChange}
                error={errors.batchSize}
              />
            </div>
            <div className="mb-4">
              <InputBox
                label={"Number of training epochs"}
                name="epochs"
                value={formData.epochs}
                onChange={handleChange}
                error={errors.epochs}
              />
            </div>
            <div className="mb-4">
              <InputBox
                label={"Train-test split ratio"}
                name="splitRatio"
                value={formData.splitRatio}
                onChange={handleChange}
                error={errors.splitRatio}
              />
            </div>

            <button
              type="submit"
              className="flex p-2 bg-[#FF52BF] w-32 rounded-lg justify-center items-center cursor-pointer text-white"
              onClick={handleSubmit}
              disabled={isSubmitting}
            >
              Train Model
            </button>
          </form>
        </div>
      </div>
    </main>
  );
}
