"use client";

import { useState, useEffect } from "react";
import axios from "axios";

import ImageClassificationSideNav from "@/components/nav/ImageClassificationSideNav";
import EpochChart from "@/components/EpochChart";
import Navbar from "@/components/nav/NavBar";
import TaskMonitor from "@/components/TaskMonitor";
import Arrow from "@/components/Arrow";

export default function Statistics({ params }) {
  const apiEndpoint =
    process.env.NEXT_PUBLIC_API_ENDPOINT + `/history/${params.projectId}/info`;

  const [historyData, setHistoryData] = useState(null);
  const [historyIndex, setHistoryIndex] = useState(0);

  useEffect(() => {
    const fetchData = async () => {
      try {
        const response = await axios.get(apiEndpoint, {
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
    fetchData();
  }, [apiEndpoint, historyIndex]);

  return (
    <main className="flex flex-col min-h-screen px-24 pb-24 bg-[#19151E] z-20">
      <Navbar />
      <div className="flex flex-row">
        <ImageClassificationSideNav params={params.projectId} />
        <div className="ml-0 lg:ml-20 mt-32">
          <div className="flex flex-row justify-between">
            <p className="text-xl text-[#FF52BF] font-bold mb-8">
              Image Classification
            </p>
            <div className="flex flex-row items-center">
              {historyIndex > 0 && <button
                onClick={() => setHistoryIndex((prevIndex) => prevIndex - 1)}
              >
                <Arrow direction="left" />
              </button>}
              {historyData && <p className="px-3">{historyIndex + 1}</p>}
              {historyIndex < historyData?.max_index && <button
                onClick={() => setHistoryIndex((prevIndex) => prevIndex + 1)}
              >
                <Arrow direction="right" />
              </button>}
            </div>
          </div>
          {!historyData && (
            <p className="font-bold mb-2">No models have been trained yet!</p>
          )}
          {historyData && (
            <div>
              <TaskMonitor resultId={historyData.history.task_id} />
              <p className="font-bold mb-2">
                {historyData.model.name} (
                {new Date(historyData.history.created_at).toLocaleString()})
              </p>
              <div className="grid grid-cols-2 gap-2">
                <div className="bg-[#3B3840] rounded-lg w-96 p-4">
                  <p className="text-white font-bold mb-2">Accuracy</p>
                  <p>{historyData.history.accuracy}</p>
                </div>
                <div className="bg-[#3B3840] rounded-lg w-96 p-4">
                  <p className="text-white font-bold mb-2">Precision</p>
                  <p>{historyData.history.precision}</p>
                </div>
                <div className="bg-[#3B3840] rounded-lg w-96 p-4">
                  <p className="text-white font-bold mb-2">Recall</p>
                  <p>{historyData.history.recall}</p>
                </div>
                <div className="bg-[#3B3840] rounded-lg w-96 p-4">
                  <p className="text-white font-bold mb-2">F1 Score</p>
                  <p>{historyData.history.f1}</p>
                </div>
              </div>
              <EpochChart epochs={historyData.epochs} />
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
                  {historyData.epochs.map((epoch, epochIndex) => (
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
          )}
        </div>
      </div>
    </main>
  );
}
