"use client";

import { useState } from "react";
import { useRouter } from "next/navigation";
import { toast } from "react-hot-toast";
import { useSelector } from "react-redux";

import Navbar from "@/components/nav/NavBar";
import FileInput from "@/components/forms/FileInput";
import CategoryInput from "@/components/forms/CategoryInput";
import InputBox from "@/components/forms/InputBox";
import axios from "axios";

export default function CreateObjectDetectionProject() {
  const router = useRouter();
  const jwtToken = useSelector((state) => state.auth.jwtToken);
  console.log(jwtToken);

  const [formData, setFormData] = useState({
    projectName: "test-objdet",
    datasetName: "test-objdet",
  });

  const [errors, setErrors] = useState({});
  const [zipFile, setTextFile] = useState(null);
  const [categoryList, setCategoryList] = useState([
    "Coverall",
    "Face_Shield",
    "Gloves",
    "Goggles",
    "Mask"
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
          project_type: "object-detection",
          class_to_label_mapping: categoryList.reduce((acc, item, index) => {
            acc[index] = item;
            return acc;
          }, {}),
        };
        console.log(payload);

        const createResponse = await axios.post(createEndpoint, payload, {
          headers: {
            "Content-Type": "application/json",
            Authorization: `Bearer ${jwtToken}`,
          },
        });

        console.log(createResponse);

        let uploadEndpoint =
          process.env.NEXT_PUBLIC_API_ENDPOINT +
          `/objdet/${createResponse.data.dataset.id}/upload`;
        const fileFormData = new FormData();
        fileFormData.append("file", zipFile);

        const uploadResponse = await axios.post(uploadEndpoint, fileFormData, {
          headers: {
            "Content-Type": "multipart/form-data",
          },
        });

        console.log(uploadResponse);

        if (createResponse.status === 201 && uploadResponse.status === 201) {
          toast.success("Job Created");
          router.push(
            `/object-detection/${createResponse.data.project.id}/label`
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
    if (!zipFile) {
      errors.zipFile = "Please select a file to upload.";
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
          <p className="text-xl text-[#D887F5] font-bold mb-8">
            Object Detection
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
              label="Upload Dataset (.ZIP only)"
              handleFileChange={handleFileChange}
              error={errors.zipFile}
              fileTypes=".zip"
            />
            <InputBox
              label="Dataset Name"
              name="datasetName"
              value={formData.datasetName}
              onChange={handleChange}
              error={errors.datasetName}
            />
            <CategoryInput
              label="Object Categories"
              categoryList={categoryList}
              setCategoryList={setCategoryList}
              error={errors.categories}
              color="bg-[#D887F5]"
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
