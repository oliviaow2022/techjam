"use client";
import { useState } from "react";
import { toast } from "react-hot-toast";
import { useRouter } from "next/navigation";

import Navbar from "@/components/nav/NavBar";
import InputBox from "@/components/forms/InputBox";
import JsonInput from "@/components/forms/JSONInput";
import RadioButton from "@/components/RadioButton";
import FileInput from "@/components/forms/FileInput";
import axios from "axios";

export default function ImageClassification() {
  const apiEndpoint = process.env.NEXT_PUBLIC_API_ENDPOINT + "/create";
  const jwtToken = localStorage.getItem("jwt");
  console.log(jwtToken)

  const [zipFile, setZipFile] = useState(null);

  const [parsedJson, setParsedJson] = useState(null);
  const handleJsonChange = (parsedJson) => {
    setParsedJson(parsedJson);
  };

  const [selectedOption, setSelectedOption] = useState("Existing dataset");
  const handleOptionChange = (e) => {
    setSelectedOption(e.target.value);
  };

  const [selectedLabelOption, setSelectedLabelOption] = useState(
    "Single Label Classification"
  );
  const handleLabelOptionChange = (e) => {
    setSelectedLabelOption(e.target.value);
  };

  const handleFileChange = (e) => {
    console.log(e.target.files[0]);
    setZipFile(e.target.files[0]);
  };

  const [formData, setFormData] = useState({
    projectName: "",
    projectType: selectedLabelOption,
    datasetName: "",
    numClasses: "",
    s3_prefix: "my-prefix/",
    classToLabelMapping: "",
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

    if (!formData.projectName) {
      errors.projectName = "Project Name is required";
    }
    if (!formData.datasetName) {
      errors.datasetName = "Dataset name is required";
    }
    if (!formData.numClasses) {
      errors.numClasses = "Number of classes is required";
    }
    if (selectedOption === "Custom Dataset" && !zipFile) {
      errors.zipFile = "Please select a file to upload.";
    }
    return errors;
  };

  const router = useRouter();

  const handleSubmit = async (e) => {
    e.preventDefault();
    const validationErrors = validate();
    setErrors(validationErrors);
    if (Object.keys(validationErrors).length === 0) {
      try {
        const createResponse = await axios.post(
          apiEndpoint,
          {
            project_name: formData.projectName,
            project_type: selectedLabelOption,
            dataset_name: formData.datasetName,
            num_classes: formData.numClasses,
            class_to_label_mapping: parsedJson,
          },
          {
            headers: {
              "Content-Type": "application/json",
              Authorization: `Bearer ${jwtToken}`,
            },
          }
        );

        console.log(createResponse);

        const zipFileFormData = new FormData();
        zipFileFormData.append("file", zipFile);

        const uploadFileResponse = await axios.post(
          `${process.env.NEXT_PUBLIC_API_ENDPOINT}/dataset/${createResponse.data.dataset.id}/upload`,
          zipFileFormData,
          {
            headers: {
              "Content-Type": "multipart/form-data",
            },
          }
        );

        console.log(uploadFileResponse.data);

        if (
          createResponse.status === 200 &&
          uploadFileResponse.status === 200
        ) {
          toast.success("Success");
        }

        router.push(`/image-classification/${createResponse.data.project.id}/label`);
        // Reset form or handle successful submission
      } catch (error) {
        console.error("Error submitting form:", error);
      } 
    }
  };

  return (
    <main className="flex flex-col min-h-screen px-24 pb-24 bg-[#19151E] z-20">
      <Navbar />
      <div className="ml-0 mt-40 w-full">
        <p
          className="text-xl text-[#FF52BF] font-bold mb-8"
          id="Project Details"
        >
          Image Classification
        </p>
        <form
          onSubmit={handleSubmit}
          className="w-full flex flex-col lg:flex-row lg:justify-between"
        >
          <div>
            <p className="font-bold mb-2">Create a new project</p>
            <div id="Model"></div>
            <div className="flex flex-col gap-4">
              <InputBox
                label={"Project Name"}
                name="projectName"
                value={formData.projectName}
                onChange={handleChange}
                error={errors.projectName}
              />
            </div>
          </div>
          <div className="bg-white border-2 mx-5 h-100% hidden lg:block"></div>
          <div className="flex flex-col gap-y-2">
            <p className="font-bold mt-12 lg:mt-0 mb-2">Dataset</p>
            <div className="flex gap-4">
              <RadioButton
                optionName={"Custom dataset"}
                selectedOption={selectedOption}
                handleOptionChange={handleOptionChange}
              />
              <RadioButton
                optionName={"Existing dataset"}
                selectedOption={selectedOption}
                handleOptionChange={handleOptionChange}
              />
            </div>
            {selectedOption === "Custom dataset" ? (
              <div>
                <InputBox
                  label={"Name of dataset"}
                  name="datasetName"
                  value={formData.datasetName}
                  onChange={handleChange}
                  error={errors.datasetName}
                />
              </div>
            ) : (
              <div>
                <InputBox
                  label={"Name of dataset"}
                  name="datasetName"
                  value={formData.datasetName}
                  onChange={handleChange}
                  error={errors.datasetName}
                />
              </div>
            )}

            <div className="flex gap-4 my-2">
              <RadioButton
                optionName={"Single Label Classification"}
                selectedOption={selectedLabelOption}
                handleOptionChange={handleLabelOptionChange}
              />
              <RadioButton
                optionName={"Multilabel Classification"}
                selectedOption={selectedLabelOption}
                handleOptionChange={handleLabelOptionChange}
              />
            </div>
            {selectedLabelOption === "Single Label Classification" ? (
              <div>
                <InputBox
                  label={"Number of classes"}
                  name="numClasses"
                  value={formData.numClasses}
                  onChange={handleChange}
                  error={errors.numClasses}
                />
              </div>
            ) : (
              <div>
                <InputBox
                  label={"Number of labels per image"}
                  name="numClasses"
                  value={formData.numClasses}
                  onChange={handleChange}
                  error={errors.numClasses}
                />
              </div>
            )}

            <JsonInput
              label={"Class to Label Mapping (JSON)"}
              name="classToLabelMapping"
              onJsonChange={handleJsonChange}
            />

            <FileInput label="Upload Dataset" handleFileChange={handleFileChange} error={errors.zipFile} fileTypes={".zip"}/>

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
