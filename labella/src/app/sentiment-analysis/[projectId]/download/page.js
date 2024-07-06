"use client";

import Navbar from "@/components/nav/NavBar";
import SentimentAnalysisSideNav from "@/components/nav/SentimentAnalysisSideNav";

export default function Download({ params }) {
  return (
    <main className="flex flex-col min-h-screen px-24 pb-24 bg-[#19151E] z-20">
      <Navbar />
      <div className="flex flex-row">
        <SentimentAnalysisSideNav params={params.projectId} />
        <form className="ml-0 lg:ml-20 mt-32">
          <p className="text-xl text-[#3FEABF] font-bold mb-8">
            Sentiment Analysis
          </p>
         
        </form>
      </div>
    </main>
  );
}
