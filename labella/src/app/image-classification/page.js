'use client';
import React, { useState } from 'react';
import Image from "next/image";
import Link from 'next/link';
import InputBox from "@/components/InputBox";
import axios from 'axios';

export default function ImageClassification() {
    const apiEndpoint = process.env.NEXT_PUBLIC_API_ENDPOINT + '/create';

    const menuOptions = ["Project Details", "Model", "Dataset", "Label", "Fine Tune", "Performance and Statistics"]
    const models = ["resnet18", "densenet121", "alexnet", "convnext_base"]

    const [selectedModel, setSelectedModel] = useState('not selected');

    const [formData, setFormData] = useState({
        projectName: '',
        projectType: '',
        model: '',
        epochs: '',
        splitRatio: '',
        batchSize: '',
        userId: 1,
        datasetName: '',
        numClasses: '',
        s3_bucket: "my-s3-bucket",
        s3_prefix: "my-prefix/",
        classToLabelMapping: {
            "0": '',
            "1": ''
        }
    });

    const [errors, setErrors] = useState({});
    const [isSubmitting, setIsSubmitting] = useState(false);

    const handleChange = (e) => {
        const { name, value } = e.target;

        if (name.startsWith("class")) {
            const classIndex = name.slice(-1); // Extract the class index from the input name
            setFormData({
                ...formData,
                classToLabelMapping: {
                    ...formData.classToLabelMapping,
                    [classIndex]: value
                }
            });
        } else {
            setFormData({
                ...formData,
                [name]: value
            });
        }
    };

    const validate = () => {
        let errors = {};

        if (!formData.projectName) {
            errors.projectName = 'Project Name is required';
        } 
        if (!formData.projectType) {
            errors.projectType = 'Project Type is required';
        }
        if (!formData.model) {
            errors.model = 'Model is required';
        }
   
        if (!formData.splitRatio) {
            errors.splitRatio = 'Train-test split ratio is required';
        }
        if (!formData.batchSize) {
            errors.batchSize = 'Batch size is required';
        }
        if (!formData.datasetName) {
            errors.datasetName = 'Dataset name is required';
        }
        if (!formData.numClasses) {
            errors.numClasses = 'Number of classes is required';
        }
        if (!formData.classToLabelMapping) {
            errors.classToLabelMapping = 'Naming of classes is required';
        }


        return errors;
    };

    const handleSubmit = async (e) => {
        e.preventDefault();
        const validationErrors = validate();
        setErrors(validationErrors);

        if (Object.keys(validationErrors).length === 0) {
            setIsSubmitting(true);
            try {
                const response = await axios.post(apiEndpoint, {
                    name: formData.projectName,
                    dataset_name: formData.datasetName,
                    model_name: formData.model_name,
                    num_classes: formData.num_classes,
                    project_name: formData.projectName,
                    project_type: formData.projectType,
                    class_to_label_mapping: formData.classToLabelMapping,
                    user_id: formData.userId,
                    s3_bucket: formData.bucket || '', // optional
                    s3_prefix: formData.prefix || '', // optional
                });

                console.log('Form submitted successfully:', response.data);
                // Reset form or handle successful submission
            } catch (error) {
                console.error('Error submitting form:', error);
            } finally {
                setIsSubmitting(false);
            }

            console.log('Form submitted successfully:', formData);

            setTimeout(() => {
                console.log('Form submitted successfully:', formData);
                setIsSubmitting(false);
            }, 1000);
        }

    };

    return (
        <main className="flex flex-col min-h-screen px-24 pb-24 bg-[#19151E] z-20">
            <div className="flex flex-row fixed top-0 h-24 w-10/12 2xl:w-full z-20 bg-[#19151E] items-end">
                <div className="z-10 max-w-5xl w-full justify-between text-sm lg:flex">
                    <Link href="/"><p className="text-xl font-bold">Labella</p></Link>
                </div>
                <div className="flex justify-around w-96">
                    <p className="mx-4">Platform</p>
                    <p className="mr-2">Datasets</p>
                    <p>Documentation</p>
                </div>
            </div>
            <div className="flex flex-row">
                <div className="flex flex-col gap-4 mr-8 pt-2 fixed top-40 z-10">
                    {menuOptions.map((option, index) => (
                        <p key={index} className="hover:cursor-pointer hover:text-[#FF52BF]">{option}</p>
                    ))}
                </div>
                <div className="bg-white border-2 mt-32 ml-64 h-screen sticky"></div>
                <div className="ml-20">
                    <p className="text-xl text-[#FF52BF] font-bold mb-8 mt-40">Image Classification</p>
                    <form onSubmit={handleSubmit}>
                        <div className="mb-4">
                            <p className="font-bold mb-4">Create a new project</p>
                            <InputBox label={"Project Name"}
                                name="projectName"
                                value={formData.projectName}
                                onChange={handleChange}
                                error={errors.projectName}
                            />
                            <InputBox label={"Project Type"}
                                name="projectType"
                                value={formData.projectType}
                                onChange={handleChange}
                                error={errors.projectType}
                            />
                        </div>

                        <div className="mb-4">
                            <p className="mt-4">Select Model</p>
                            <div className="grid grid-cols-4 gap-6">
                                {models.map((model, index) => (
                                    <div key={index} className={`flex border w-40 2xl:w-64 items-center justify-center rounded-lg h-8 cursor-pointer ${selectedModel === model ? 'bg-[#FF52BF] text-black' : 'hover:bg-[#FF52BF] hover:text-black'}`}
                                        onClick={() => { setFormData({ ...formData, model }); setSelectedModel(model) }}>
                                        {model}
                                    </div>
                                ))}
                            </div>
                            {errors.model && <p className="text-red-500">{errors.model}</p>}
                        </div>

                        <div className="mb-4">
                            <InputBox label={"Number of training epochs"}
                                name="epochs"
                                value={formData.epochs}
                                onChange={handleChange}
                                error={errors.epochs}
                            />
                        </div>

                        <div className="mb-4">
                            <InputBox label={"Train-test split ratio"}
                                name="splitRatio"
                                value={formData.splitRatio}
                                onChange={handleChange}
                                error={errors.splitRatio}
                            />
                        </div>

                        <div className="mb-4">
                            <InputBox label={"Batch size"}
                                name="batchSize"
                                value={formData.batchSize}
                                onChange={handleChange}
                                error={errors.batchSize}
                            />
                        </div>


                        <div className="mb-4">
                            {/* <p className="">Choose Dataset</p>
                            <div className="flex border w-64 rounded-lg pl-4 py-1 items-center">
                                <img src="/upload.png" alt="upload icon" className="h-4 mr-2"></img>
                                <p>Upload Dataset</p>
                            </div> */}
                            <InputBox label={"Name of dataset"}
                                name="datasetName"
                                value={formData.datasetName}
                                onChange={handleChange}
                                error={errors.datasetName}
                            />
                        </div>
                        <div className="mb-4">
                            <InputBox label={"Number of classes"}
                                name="numClasses"
                                value={formData.numClasses}
                                onChange={handleChange}
                                error={errors.numClasses}
                            />
                            <InputBox label={"Name of class 0"}
                                name="class0"
                                value={formData.classToLabelMapping[0]}
                                onChange={handleChange}
                                error={errors.classToLabelMapping}
                            />
                            <InputBox label={"Name of class 1"}
                                name="class1"
                                value={formData.classToLabelMapping[1]}
                                onChange={handleChange}
                                error={errors.classToLabelMapping}
                            />
                        </div>
                        <div>
                            <button type="submit" className="flex bg-[#FF52BF] w-32 rounded-full justify-center items-center cursor-pointer" disabled={isSubmitting}>
                                {isSubmitting ? 'Creating...' : 'Create Project'}
                            </button>
                        </div>

                    </form>

                </div>
            </div>


        </main>
    );
}
