import Image from "next/image";
import Link from 'next/link';

export default function Home() {
    return (
        <main className="flex flex-col min-h-screen p-24">
            <div className="flex flex-row mb-10">
                <div className="z-10 max-w-5xl w-full justify-between text-sm lg:flex">
                    <Link href="/"><p className="text-xl font-bold">Labella</p></Link>
                </div>
                <div className="flex justify-around w-96">
                    <p className="mx-4">Platform</p>
                    <p className="mr-2">Datasets</p>
                    <p>Documentation</p>
                </div>
            </div>
            <div className="flex justify-center">
                <p className="text-md font-bold mb-4">Discover Our Data Labeling Solutions</p>
            </div>
            <div className="flex flex-col gap-8">
                <div className="flex justify-center gap-8">
                    <div className="bg-[#3FEABF] h-56 w-96 rounded-xl text-black flex items-center justify-center flex-col">
                        <Image
                            src="/sentiment analysis.png"
                            width={150}
                            height={150}
                            alt="sentiment analysis icon"
                        />
                        <p className="mt-2">Sentiment Analysis</p>
                    </div>
                    <div className="bg-[#FF52BF] h-56 w-96 rounded-xl text-black flex items-center justify-center flex-col">
                        <Image
                            src="/image classification.png"
                            width={150}
                            height={150}
                            alt="sentiment analysis icon"
                        />
                        <p>Image Classification</p>
                    </div>
                </div>
                <div className="flex justify-center gap-8">
                    <div className="bg-[#D887F5] h-56 w-96 rounded-xl text-black flex items-center justify-center flex-col">
                        <Image
                            src="/object detection.png"
                            width={130}
                            height={130}
                            alt="sentiment analysis icon"
                        />
                        <p className="mt-4">Object Detection</p>
                    </div>
                    <div className="bg-[#1ED2EC] h-56 w-96 rounded-xl text-black flex items-center justify-center flex-col">
                        <Image
                            src="/information extraction.png"
                            width={130}
                            height={130}
                            alt="sentiment analysis icon"
                        />
                        <p className="mt-4">Information Extraction</p>
                    </div>
                </div>
            </div>
        </main>
    );
}
