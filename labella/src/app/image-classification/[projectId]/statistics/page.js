"use client";

import { useState, useEffect } from "react";
import { toast } from "react-hot-toast";

import ImageClassificationSideNav from "@/components/nav/ImageClassificationSideNav";
import EpochChart from "@/components/EpochChart";
import Navbar from "@/components/nav/NavBar";
import axios from "axios";

export default function Statistics({ params }) {
  const apiEndpoint =
    process.env.NEXT_PUBLIC_API_ENDPOINT + `/history/${params.projectId}/info`;
  const jwtToken = localStorage.getItem("jwt");
  const config = {
    headers: {
      "Content-Type": "application/json",
      Authorization: `Bearer ${jwtToken}`,
    },
  };
  const [modelTrainHistory, setModelTrainHistory] = useState([]);
  useEffect(() => {
    const fetchData = async () => {
      try {
        const response = await axios.get(apiEndpoint);
        console.log(response.data);
        setModelTrainHistory(response.data);
      } catch (error) {
        console.error(error.message);
      }
    };
    fetchData();
  }, [apiEndpoint]);

  const handleDownloadModel = async (modelId) => {
    try {
      toast.success("Downloading file");
      let response = await axios.get(
        process.env.NEXT_PUBLIC_API_ENDPOINT +
          `/model/${modelId}/download`,
        {
          responseType: "blob", // Important: 'blob' indicates binary data
        }
      );

      console.log(response)
      if (response.status === 200) {
        const url = window.URL.createObjectURL(new Blob([response.data]));
        const link = document.createElement('a');
        link.href = url;
        link.setAttribute('download', `model.pth`); // Set the file name here
        document.body.appendChild(link);
        link.click();
        document.body.removeChild(link);
      }
    } catch (error) {
      console.log(error)
    }
   
  };

  return (
    <main className="flex flex-col min-h-screen px-24 pb-24 bg-[#19151E] z-20">
      <Navbar />
      <div className="flex flex-row">
        <ImageClassificationSideNav params={params.projectId} />
        <div className="ml-0 lg:ml-20 mt-32">
          <p className="text-xl text-[#FF52BF] font-bold mb-8">
            Image Classification
          </p>
          {modelTrainHistory.length == 0 && (
            <p className="font-bold mb-2">No models have been trained yet!</p>
          )}
          {modelTrainHistory.map((trainHistory) => (
            <div>
              <div className="flex flex-row justify-between mb-2">
                <p className="font-bold mb-2">
                  {trainHistory.model.name} (
                  {new Date(trainHistory.history.created_at).toLocaleString()})
                </p>
                <button
                  className="flex py-2 px-4 bg-[#FF52BF] w-fit rounded-lg justify-center items-center cursor-pointer text-white"
                  onClick={() => handleDownloadModel(trainHistory.model.id)}
                >
                  Download Model
                </button>
              </div>
              <div className="grid grid-cols-2 gap-2">
                <div className="bg-[#3B3840] rounded-lg p-4">
                  <p className="text-white font-bold mb-2">Accuracy</p>
                  <p>{trainHistory.history.accuracy}</p>
                </div>
                <div className="bg-[#3B3840] rounded-lg  p-4">
                  <p className="text-white font-bold mb-2">Precision</p>
                  <p>{trainHistory.history.precision}</p>
                </div>
                <div className="bg-[#3B3840] rounded-lg p-4">
                  <p className="text-white font-bold mb-2">Recall</p>
                  <p>{trainHistory.history.recall}</p>
                </div>
                <div className="bg-[#3B3840] rounded-lg p-4">
                  <p className="text-white font-bold mb-2">F1 Score</p>
                  <p>{trainHistory.history.f1}</p>
                </div>
              </div>
              <EpochChart epochs={trainHistory.epochs} />
              <table className="table-auto w-full border-collapse border border-slate-500 my-5">
                <thead>
                  <tr>
                    <th className="px-4 py-2 border border-slate-600">Epoch</th>
                    <th className="px-4 py-2 border border-slate-600">
                      Train Accuracy
                    </th>
                    <th className="px-4 py-2 border border-slate-600">
                      Train Loss
                    </th>
                    <th className="px-4 py-2 border border-slate-600">
                      Validation Accuracy
                    </th>
                    <th className="px-4 py-2 border border-slate-600">
                      Validation Loss
                    </th>
                  </tr>
                </thead>
                <tbody>
                  {trainHistory.epochs.map((epoch, epochIndex) => (
                    <tr key={epochIndex}>
                      <td className="border px-4 py-2 border-slate-700">
                        {epochIndex + 1}
                      </td>
                      <td className="border px-4 py-2 border-slate-700">
                        {epoch.train_acc}
                      </td>
                      <td className="border px-4 py-2 border-slate-700">
                        {epoch.train_loss}
                      </td>
                      <td className="border px-4 py-2 border-slate-700">
                        {epoch.val_acc}
                      </td>
                      <td className="border px-4 py-2 border-slate-700">
                        {epoch.val_loss}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          ))}
        </div>
      </div>
    </main>
  );
}
