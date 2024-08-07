"use client";

import { useState, useEffect } from "react";
import { toast } from "react-hot-toast";
import createApiClient from "@/components/axiosInstance";

import ObjectDetectionSideNav from "@/components/nav/ObjectDetectionSideNav";
import EpochChart from "@/components/EpochChart";
import Navbar from "@/components/nav/NavBar";
import TaskMonitor from "@/components/TaskMonitor";
import Arrow from "@/components/Arrow";

export default function ObjectDetectionStatistics({ params }) {
  const apiClient = createApiClient();
  const apiEndpoint =
    process.env.NEXT_PUBLIC_API_ENDPOINT + `/history/${params.projectId}/info`;

  const [historyData, setHistoryData] = useState(null);
  const [historyIndex, setHistoryIndex] = useState(0);

  const fetchData = async () => {
    try {
      const response = await apiClient.get(apiEndpoint, {
        params: {
          index: historyIndex,
        },
      });

      console.log(response.data);
      setHistoryData(response.data);
    } catch (error) {
      console.error(error.message);
    }
  };

  useEffect(() => {
    fetchData();
  }, [apiEndpoint, historyIndex]);

  const handleDownloadModel = async (historyId) => {
    try {
      toast.success("Downloading file");
      let response = await apiClient.get(
        process.env.NEXT_PUBLIC_API_ENDPOINT + `/model/${historyId}/download`,
        {
          responseType: "blob", // Important: 'blob' indicates binary data
        }
      );

      console.log(response);
      if (response.status === 200) {
        const url = window.URL.createObjectURL(new Blob([response.data]));
        const link = document.createElement("a");
        link.href = url;
        link.setAttribute("download", `model.pt`); // Set the file name here
        document.body.appendChild(link);
        link.click();
        document.body.removeChild(link);
      }
    } catch (error) {
      console.log(error);
    }
  };

  const handleDownloadDataset = async (projectId) => {
    try {
      // Make GET request to backend endpoint
      const response = await apiClient.get(
        process.env.NEXT_PUBLIC_API_ENDPOINT + `/objdet/${projectId}/download`,
        {
          responseType: "blob", // Important to handle file download
        }
      );

      // Create a URL for the downloaded file
      const url = window.URL.createObjectURL(new Blob([response.data]));
      const link = document.createElement("a");
      link.href = url;
      link.setAttribute("download", "annotations.csv"); // Specify file name
      document.body.appendChild(link);
      link.click();
      link.remove();
    } catch (error) {
      console.error("Error downloading CSV:", error);
    }
  };

  const handleTaskSuccess = () => {
    fetchData(); // Fetch data when the task is successful
  };

  return (
    <main className="flex flex-col min-h-screen px-24 pb-24 bg-[#19151E] z-20">
      <Navbar />
      <div className="flex flex-row">
        <ObjectDetectionSideNav params={params.projectId} />
        <div className="ml-0 lg:ml-20 mt-32 w-full">
          <div className="flex flex-row justify-between">
            <p className="text-xl text-[#D887F5] font-bold mb-8">
              Object Detection
            </p>
            {historyData && (
              <div className="flex flex-row items-center">
                {historyIndex > 0 && (
                  <button
                    onClick={() =>
                      setHistoryIndex((prevIndex) => prevIndex - 1)
                    }
                  >
                    <Arrow direction="left" />
                  </button>
                )}
                <p className="px-3">{historyIndex + 1}</p>
                {historyIndex < historyData?.max_index - 1 && (
                  <button
                    onClick={() =>
                      setHistoryIndex((prevIndex) => prevIndex + 1)
                    }
                  >
                    <Arrow direction="right" />
                  </button>
                )}
              </div>
            )}
          </div>
          {!historyData && (
            <p className="font-bold mb-2">No models have been trained yet!</p>
          )}
          {historyData && (
            <div>
              <TaskMonitor
                resultId={historyData.history?.task_id}
                onSuccess={handleTaskSuccess}
              />
              <div className="flex flex-row gap-2 mb-8">
                <button
                  className="flex py-2 px-4 bg-[#FF52BF] w-fit rounded-lg justify-center items-center cursor-pointer text-white disabled:opacity-75"
                  onClick={() => handleDownloadModel(historyData.history?.id)}
                  disabled={!historyData?.history?.model_path}
                >
                  Download Model
                </button>
                <button
                  className="flex py-2 px-4 bg-[#FF52BF] w-fit rounded-lg justify-center items-center cursor-pointer text-white disabled:opacity-75"
                  onClick={() => handleDownloadDataset(params.projectId)}
                  disabled={!historyData?.history?.model_path}
                >
                  Download Dataset
                </button>
              </div>
              <p className="font-bold mb-2">
                {historyData.model?.name} (
                {new Date(historyData.history?.created_at).toLocaleString()})
              </p>
              <div className="grid grid-cols-2 gap-2">
                <div className="bg-[#3B3840] rounded-lg  p-4">
                  <p className="text-white font-bold mb-2">Precision</p>
                  <p>{historyData.history?.precision}</p>
                </div>
                <div className="bg-[#3B3840] rounded-lg p-4">
                  <p className="text-white font-bold mb-2">Recall</p>
                  <p>{historyData.history?.recall}</p>
                </div>
              </div>
              {historyData.epochs && <EpochChart epochs={historyData.epochs} />}
              <table className="table-auto w-full border-collapse border border-slate-500 my-5">
                <thead>
                  <tr>
                    <th className="px-4 py-2 border border-slate-600">Epoch</th>
                    <th className="px-4 py-2 border border-slate-600">
                      Train Loss
                    </th>
                    <th className="px-4 py-2 border border-slate-600">
                     Precision
                    </th>
                    <th className="px-4 py-2 border border-slate-600">
                      Recall
                    </th>
                  </tr>
                </thead>
                <tbody>
                  {historyData.epochs?.map((epoch, epochIndex) => (
                    <tr key={epochIndex}>
                      <td className="border px-4 py-2 border-slate-700">
                        {epochIndex + 1}
                      </td>
                      <td className="border px-4 py-2 border-slate-700">
                        {epoch.train_loss}
                      </td>
                      <td className="border px-4 py-2 border-slate-700">
                        {epoch.precision}
                      </td>
                      <td className="border px-4 py-2 border-slate-700">
                        {epoch.recall}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </div>
      </div>
    </main>
  );
}
