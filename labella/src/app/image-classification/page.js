"use client";
import React, { useState } from "react";
import Navbar from "@/components/NavBar";
import InputBox from "@/components/InputBox";
import JsonInput from "@/components/JSONInput";
import RadioButton from "@/components/RadioButton";
import SideNav from "@/components/SideNav";
import axios from "axios";
import { useRouter } from "next/navigation";

export default function ImageClassification() {
  const apiEndpoint = process.env.NEXT_PUBLIC_API_ENDPOINT + "/create";
  const userId = localStorage.getItem("user_id");
  const jwtToken = localStorage.getItem("jwt");

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

  const [projectCreated, setProjectCreated] = useState(false);

  const config = {
    headers: {
      "Content-Type": "application/json",
      Authorization: `Bearer ${jwtToken}`,
    },
  };
  const [formData, setFormData] = useState({
    projectName: "",
    projectType: "image-classification",
    userId: userId,
    datasetName: "",
    numClasses: "",
    s3_bucket: "my-s3-bucket",
    s3_prefix: "my-prefix/",
    classToLabelMapping: "",
    config: config,
  });

  const [errors, setErrors] = useState({});
  const [isSubmitting, setIsSubmitting] = useState(false);

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
    return errors;
  };

  const router = useRouter();

  const handleSubmit = async (e) => {
    e.preventDefault();
    const validationErrors = validate();
    setErrors(validationErrors);
    if (Object.keys(validationErrors).length === 0) {
      setIsSubmitting(true);
      try {
        const response = await axios.post(
          apiEndpoint,
          {
            name: formData.projectName,
            dataset_name: formData.datasetName,
            num_classes: formData.numClasses,
            project_name: formData.projectName,
            project_type: formData.projectType,
            class_to_label_mapping: parsedJson,
            user_id: formData.userId,
            s3_bucket: formData.bucket || "", // optional
            s3_prefix: formData.prefix || "", // optional
          },
          formData.config
        );

        console.log("Form submitted successfully:", response.data);
        setProjectCreated(true);
        router.push(`/label/${response.data.project.id}`);
        // Reset form or handle successful submission
      } catch (error) {
        console.error("Error submitting form:", error);
      } finally {
        setIsSubmitting(false);
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
          <div className="bg-white border-2 ml-64 h-100% hidden lg:block"></div>
          <div>
            <p className="font-bold mt-12 lg:mt-0 mb-2">Dataset</p>
            <div className="flex gap-4 mb-4">
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

            <div className="flex gap-4">
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

            <button
              type="submit"
              className="flex my-4 p-2 bg-[#FF52BF] w-32 rounded-lg justify-center items-center cursor-pointer text-white"
              disabled={isSubmitting}
            >
              {isSubmitting ? "Creating..." : "Create Project"}
            </button>
          </div>
          {projectCreated ? <div>Success!</div> : <div></div>}
        </form>
      </div>
    </main>
  );
}
