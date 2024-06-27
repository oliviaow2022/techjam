"use client";
import React, { useState, useEffect } from "react";
import Image from "next/image";
import Link from "next/link";
import LabelButton from "@/components/LabelButton";
import ImageSlider from "@/components/ImageSlider";
import SideNav from "@/components/SideNav";
import axios from "axios";
import { useRouter } from "next/navigation";
import { toast } from "react-hot-toast";

export default function Label({ params }) {
  const batchApiEndpoint =
    process.env.NEXT_PUBLIC_API_ENDPOINT + `/dataset/${params.id}/batch`;
  const datasetApiEndpoint =
    process.env.NEXT_PUBLIC_API_ENDPOINT + `/dataset/${params.id}`;
  const jwtToken = localStorage.getItem("jwt");

  const config = {
    headers: {
      "Content-Type": "application/json",
      Authorization: `Bearer ${jwtToken}`,
    },
  };

  const [images, setImages] = useState([]);
  const [datasetData, setDatasetData] = useState({});
  const [error, setError] = useState(null);
  const [loading, setLoading] = useState(true);
  const [currentIndex, setCurrentIndex] = useState(0);

  useEffect(() => {
    const fetchData = async () => {
      try {
        const batchResponse = await axios.get(batchApiEndpoint);
        setImages(batchResponse.data);

        const datasetResponse = await axios.get(datasetApiEndpoint);
        setDatasetData(datasetResponse.data);
      } catch (error) {
        setError(error.message);
      } finally {
        setLoading(false);
      }
    };

    fetchData();
  }, [batchApiEndpoint, datasetApiEndpoint]);

  if (loading) {
    return <div>Loading...</div>;
  }
  if (error) {
    return <div>Error: {error}</div>;
  }

  const handleImageChange = (idx) => {
    setCurrentIndex(idx);
  };

  const handleLabelAddition = async (classInteger) => {
    let apiEndpoint =
      process.env.NEXT_PUBLIC_API_ENDPOINT +
      `/instance/${images[currentIndex].id}/set_label`;
    try {
      const response = await axios.post(apiEndpoint, {
        labels: classInteger,
      });
      if (response.status === 200) {
        setCurrentIndex((currentIndex) => (currentIndex += 1));
        toast.success("Label updated")
      }
    } catch (err) {
      console.log(err);
    }
  };

  return (
    <main className="flex flex-col min-h-screen px-24 pb-24 bg-[#19151E] z-20">
      <div className="flex flex-row fixed top-0 h-24 w-10/12 2xl:w-full z-20 bg-[#19151E] items-end">
        <div className="z-10 max-w-5xl w-full justify-between text-sm lg:flex">
          <Link href="/">
            <p className="text-xl font-bold">Labella</p>
          </Link>
        </div>
        <div className="flex justify-around w-96">
          <p className="mx-4">Platform</p>
          <p className="mr-2">Datasets</p>
          <p>Documentation</p>
        </div>
      </div>
      <div className="flex flex-row">
        <SideNav params={params.id} />
        <div className="bg-white border-2 mt-32 ml-64 h-100% hidden lg:block"></div>
        <div className="ml-0 lg:ml-20">
          <p className="text-xl text-[#FF52BF] font-bold mb-8 mt-40">
            Image Classification
          </p>
          <p className="font-bold mb-2">Label Images</p>
          <div className="flex gap-4 mb-4 items-center justify-center">
            <ImageSlider
              images={images}
              bucketname={"dltechjam"}
              bucketprefix={"transfer-antsbees"}
              handleImageChange={handleImageChange}
              currentIndex={currentIndex}
              setCurrentIndex={setCurrentIndex}
            />
            <div className="bg-[#3B3840] rounded-lg w-96 p-4">
              <p className="text-white font-bold mb-2">Class</p>
              <div className="flex gap-4">
                {Object.entries(datasetData.class_to_label_mapping).map(
                  ([key, value]) => (
                    <LabelButton
                      key={key}
                      classInteger={key}
                      name={value}
                      handleOptionChange={handleLabelAddition}
                    />
                  )
                )}
              </div>
            </div>
          </div>
          <div className="grid grid-cols-10 gap-2">
            {images.map((item, index) => {
              return (
                <div key={item.data} onClick={() => setCurrentIndex(index)}>
                  <img
                    src={`https://dltechjam.s3.amazonaws.com/transfer-antsbees/${item.data}`}
                    className={`w-20 h-20 rounded-lg ${
                      currentIndex === index ? "border-4 border-white" : ""
                    }`}
                  />
                  <p className="break-all" style={{ fontSize: "10px" }}>
                    {item.data}
                  </p>
                </div>
              );
            })}
          </div>
        </div>
      </div>
    </main>
  );
}
