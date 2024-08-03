"use client";

import Image from "next/image";
import Link from "next/link";
import Navbar from "@/components/nav/NavBar";
import { useState, useEffect } from "react";
import { useRouter } from "next/navigation";
import axios from "axios";

export default function Home() {
  const userId = localStorage.getItem("user_id");
  const router = useRouter();

  const apiEndpoint =
    process.env.NEXT_PUBLIC_API_ENDPOINT + `/user/${userId}/projects`;
  const [userProjects, setUserProjects] = useState([]);

  useEffect(() => {
    const fetchUserProjects = async () => {
      try {
        const response = await axios.get(apiEndpoint);
        console.log(response.data);
        setUserProjects(response.data);
      } catch (err) {
        console.log(err);
      }
    };

    fetchUserProjects(), [];
  });

  const redirect = (project) => {
    if (project.type === "Single Label Classification" || project.type === "Multilabel Classification") {
      router.push(`/image-classification/${project.id}/label`);
    } else {
      router.push(`/sentiment-analysis/${project.id}/label`);
    }
  };

  return (
    <main className="min-h-screen px-24 bg-[#19151E] z-20">
      <Navbar />
      <div className="flex flex-col pt-24 justify-center h-screen">
        {userProjects.length > 0 && (
          <div className="overflow-x-auto">
            <p className="text-white font-bold mb-2">Your Existing Projects</p>
            <div className="flex gap-8">
              {userProjects.map((project) => (
                <div
                  className="border border-white h-56 w-64 rounded-xl flex flex-col justify-between cursor-pointer hover:bg-white hover:text-black p-5"
                  onClick={() => redirect(project)}
                >
                  <p>{project.name}</p>
                  {(project.type === "Single Label Classification" || project.type === "Multilabel Classification") ? (
                    <div className="flex flex-row items-center">
                      <div className="w-3 h-3 rounded-lg bg-[#FF52BF] mr-2" />
                      Image Classification
                    </div>
                  ) : (
                    <div className="flex flex-row items-center">
                      <div className="w-3 h-3 rounded-lg bg-[#3FEABF] mr-2" />
                      Sentiment Analysis
                    </div>
                  )}
                </div>
              ))}
            </div>
            <hr className="my-4" />
          </div>
        )}
        <p className="text-white font-bold mb-2">Create a new project</p>
        <div className="flex gap-8">
          <Link href="/sentiment-analysis/create">
            <div className="bg-[#3FEABF] h-56 w-64 rounded-xl text-black flex items-center justify-center flex-col cursor-pointer hover:opacity-90 active:opacity-100">
              <Image
                src="/sentiment analysis.png"
                width={150}
                height={150}
                alt="sentiment analysis icon"
              />
              <p className="mt-2">Sentiment Analysis</p>
            </div>
          </Link>
          <Link href="/image-classification/create">
            <div className="bg-[#FF52BF] h-56 w-64 rounded-xl text-black flex items-center justify-center flex-col cursor-pointer hover:opacity-90 active:opacity-100">
              <Image
                src="/image classification.png"
                width={150}
                height={150}
                alt="sentiment analysis icon"
              />
              <p>Image Classification</p>
            </div>
          </Link>

          <Link href="/object-detection/create">
            <div className="bg-[#D887F5] h-56 w-64 rounded-xl text-black flex items-center justify-center flex-col cursor-pointer hover:opacity-90 active:opacity-100">
              <Image
                src="/object detection.png"
                width={130}
                height={130}
                alt="sentiment analysis icon"
              />
              <p className="mt-4">Object Detection</p>
            </div>
          </Link>
        </div>
      </div>
    </main>
  );
}
