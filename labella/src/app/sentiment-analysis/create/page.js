"use client";

import { useState } from "react";
import { useRouter } from "next/navigation";
import { toast } from "react-hot-toast";
import { useAppSelector } from "@/store/store";

import Navbar from "@/components/nav/NavBar";
import FileInput from "@/components/forms/FileInput";
import CategoryInput from "@/components/forms/CategoryInput";
import InputBox from "@/components/forms/InputBox";
import axios from "axios";

export default function SentimentAnalysis() {
  const router = useRouter();
  const jwtToken = useAppSelector((state) => state.auth.jwtToken);
  console.log(jwtToken)

  const [formData, setFormData] = useState({
    projectName: "test-senti",
    datasetName: "test-senti",
    textColumn: "Text",
  });

  const [errors, setErrors] = useState({});
  const [textFile, setTextFile] = useState(null);
  const [categoryList, setCategoryList] = useState([
    "positive",
    "negative",
    "neutral",
  ]);

  const handleFileChange = (e) => {
    console.log(e.target.files[0]);
    setTextFile(e.target.files[0]);
  };

  const handleChange = (e) => {
    const { name, value } = e.target;

    setFormData({
      ...formData,
      [name]: value,
    });
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    let validationErrors = validate();
    setErrors(validationErrors);

    if (Object.keys(validationErrors).length === 0) {
      try {
        const createEndpoint = process.env.NEXT_PUBLIC_API_ENDPOINT + "/create";

        const payload = {
          project_name: formData.projectName,
          num_classes: categoryList.length,
          dataset_name: formData.datasetName,
          project_type: "sentiment-analysis",
          class_to_label_mapping: categoryList.reduce((acc, item, index) => {
            acc[index] = item;
            return acc;
          }, {}),
        };
        console.log(payload);

        // const jwtToken = localStorage.getItem("jwt");
        console.log("Token from slice:", jwtToken);

        const createResponse = await axios.post(createEndpoint, payload, {
          headers: {
            "Content-Type": "application/json",
            Authorization: `Bearer ${jwtToken}`,
          },
        });

        console.log(createResponse);

        let uploadEndpoint =
          process.env.NEXT_PUBLIC_API_ENDPOINT +
          `/senti/${createResponse.data.dataset.id}/upload`;
        const fileFormData = new FormData();
        fileFormData.append("file", textFile);
        fileFormData.append("text_column", formData.textColumn);

        const uploadResponse = await axios.post(uploadEndpoint, fileFormData, {
          headers: {
            "Content-Type": "multipart/form-data",
          },
        });

        console.log(uploadResponse);

        if (createResponse.status === 201 && uploadEndpoint.status === 200) {
          toast.success("Success");
          router.push(
            `/sentiment-analysis/${createResponse.data.project.id}/label`
          );
        }
      } catch (error) {
        console.log(error);
      }
    }
  };

  const validate = () => {
    let errors = {};

    if (!categoryList) {
      errors.categoryList = "Missing sentiment categories";
    }
    if (!textFile) {
      errors.textFile = "Please select a file to upload.";
    }
    if (!formData.projectName) {
      errors.projectName = "Project name is required";
    }
    return errors;
  };

  return (
    <main className="flex flex-col min-h-screen px-24 pb-24 bg-[#19151E] z-20">
      <Navbar />
      <div className="flex flex-row">
        <form className="ml-0 lg:ml-20 mt-32" onSubmit={handleSubmit}>
          <p className="text-xl text-[#3FEABF] font-bold mb-8">
            Sentiment Analysis
          </p>
          <div className="flex flex-col gap-y-5">
            <InputBox
              label="Project Name"
              name="projectName"
              value={formData.projectName}
              onChange={handleChange}
              error={errors.projectName}
            />

            <FileInput
              label="Upload Dataset (CSV/XLSX only)"
              handleFileChange={handleFileChange}
              error={errors.textFile}
              fileTypes=".csv, .txt"
            />
            <InputBox
              label="Dataset Name"
              name="datasetName"
              value={formData.datasetName}
              onChange={handleChange}
              error={errors.datasetName}
            />
            <CategoryInput
              label="Sentiment Categories"
              categoryList={categoryList}
              setCategoryList={setCategoryList}
              error={errors.categories}
              color="bg-[#3FEABF]"
            />
            <InputBox
              label="Text Column in Dataframe"
              name="textColumn"
              value={formData.textColumn}
              onChange={handleChange}
              error={errors.textColumn}
            />

            <button
              type="submit"
              className="flex my-4 py-2 px-4 bg-[#FF52BF] w-fit rounded-lg justify-center items-center cursor-pointer text-white"
            >
              Create Project
            </button>
          </div>
        </form>
      </div>
    </main>
  );
}
