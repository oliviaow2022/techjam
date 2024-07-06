"use client";

import { toast } from 'react-hot-toast';

import Navbar from "@/components/nav/NavBar";
import SentimentAnalysisSideNav from "@/components/nav/SentimentAnalysisSideNav";
import LabelButton from "@/components/LabelButton";

const datasetData = {
  class_to_label_mapping: {
    0: "negative",
    1: "positive",
    2: "neutral",
  },
};

export default function ManualLabelling({ params }) {
  const handleLabelAddition = async (classInteger) => {
    toast.success('Label updated')
  };

  return (
    <main className="flex flex-col min-h-screen px-24 pb-24 bg-[#19151E] z-20">
      <Navbar />
      <div className="flex flex-row">
        <SentimentAnalysisSideNav params={params.projectId} />
        <form className="ml-0 lg:ml-20 mt-32">
          <p className="text-xl text-[#3FEABF] font-bold mb-8">
            Sentiment Analysis
          </p>
          <div className="flex flex-row gap-4">
            <div className="w-96 h-64 rounded-lg border-2 border-white p-4">
              <p className="text-white font-bold mb-2">Data</p>
              Just finished an amazing workout! 💪
            </div>

            <div className="bg-[#3B3840] rounded-lg w-96 p-4">
              <p className="text-white font-bold mb-2">Class</p>
              <div className="flex flex-wrap justify-between">
                {Object.entries(datasetData?.class_to_label_mapping).map(
                  ([key, value]) => (
                    <LabelButton
                      key={key}
                      classInteger={key}
                      name={value}
                      handleOptionChange={handleLabelAddition}
                      bgColour="bg-[#3FEABF]"
                    />
                  )
                )}
              </div>
            </div>
          </div>
        </form>
      </div>
    </main>
  );
}
