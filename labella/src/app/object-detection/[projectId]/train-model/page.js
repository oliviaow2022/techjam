"use client";

import { useState } from "react";
import { toast } from "react-hot-toast";
import { useRouter } from "next/navigation";

import ObjectDetectionSideNav from "@/components/nav/ObjectDetectionSideNav";
import Navbar from "@/components/nav/NavBar";
import InputBox from "@/components/forms/InputBox";
import axios from "axios"


const models = [
  {
    name: "Faster R-CNN ResNet-50 FPN",
    description:
      "This model combines the strengths of Faster R-CNN, ResNet-50, and Feature Pyramid Network (FPN) architectures.",
  },
];

export default function ObjectDetectionTrainModel({ params }) {
  const router = useRouter();

  const [formData, setFormData] = useState({
    batch_size: 128,
    num_epochs: 3,
    train_test_split: 0.8,
    model_name: "",
    model_description: "",
  });
  const [errors, setErrors] = useState({});

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

  const handleSubmit = async (e) => {
    e.preventDefault();

    let validationErrors = validate();
    setErrors(validationErrors);

    if (Object.keys(validationErrors).length === 0) {
      try {
        let apiEndpoint =
          process.env.NEXT_PUBLIC_API_ENDPOINT +
          `/objdet/${params.projectId}/train`;
        console.log(formData);
        const response = await axios.post(apiEndpoint, formData);

        console.log(response);

        if (response.status === 200) {
          toast.success(`Job ID ${response.data.task_id} created`);
          router.push(`/object-detection/${params.projectId}/statistics`)
        }
      } catch (err) {
        console.log(err);
      }
    }
  };

  return (
    <main className="flex flex-col min-h-screen px-24 pb-24 bg-[#19151E] z-20">
      <Navbar />
      <div className="flex flex-row">
        <ObjectDetectionSideNav params={params.projectId} />
        <div className="ml-0 lg:ml-20">
          <p
            className="text-xl text-[#D887F5] font-bold mb-8 mt-40"
            id="Train Model"
          >
            Train Model
          </p>
          <form>
            <p className="mt-2">Select Model Architecture</p>
            <div className="mb-4 flex flex-row flex-wrap gap-x-4 gap-y-1">
            {models.map((model, index) => (
                <div
                  key={index}
                  className={`border border-white border-opacity-50 w-72 rounded-lg cursor-pointer my-1 p-4 ${
                    formData.model_name === model.name
                      ? "bg-[#D887F5] text-black"
                      : "hover:bg-[#D887F5] hover:text-black"
                  }`}
                  onClick={() => {
                    setFormData({
                      ...formData,
                      model_name: model.name,
                      model_description: model.description,
                    });
                  }}
                >
                  <p className="font-bold mb-2">{model.name}</p>
                  <p>{model.description}</p>
                </div>
              ))}
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
