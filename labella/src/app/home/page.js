import Image from "next/image";
import Link from 'next/link';

export default function Home() {
    const isLoggedIn = () => {
        const token = localStorage.getItem('jwt');
        return token !== null;
      };
    return (
        <main className="flex flex-col min-h-screen px-24 pb-24 bg-[#19151E] z-20">
            <div className="flex flex-row fixed top-0 h-24 w-10/12 2xl:w-full z-20 bg-[#19151E] items-end">
                <div className="z-10 max-w-5xl w-full justify-between text-sm lg:flex">
                    <Link href="/"><p className="text-xl font-bold">Labella</p></Link>
                </div>
                <div className="flex justify-around w-96 items-center">
                    <p className="mx-4 cursor-pointer">Platform</p>
                    <p className="mr-2 cursor-pointer">Datasets</p>
                    <p className="mr-2 cursor-pointer">Documentation</p>
                    {isLoggedIn ? (<Image
                        src="/profileicon.png"
                        width={30}
                        height={15}
                        alt="profile icon"
                        className="cursor-pointer"
                    />) : (<div></div>)}
                    
                </div>
            </div>
            <div className="flex flex-col justify-center items-center h-screen">
                <div className="flex justify-center">
                    <p className="text-md font-bold mb-4">Discover Our Data Labeling Solutions</p>
                </div>
                <div className="flex flex-col gap-8">
                    <div className="flex justify-center gap-8">
                        <div className="bg-[#3FEABF] h-56 w-96 rounded-xl text-black flex items-center justify-center flex-col cursor-pointer hover:opacity-90 active:opacity-100">
                            <Image
                                src="/sentiment analysis.png"
                                width={150}
                                height={150}
                                alt="sentiment analysis icon"
                            />
                            <p className="mt-2">Sentiment Analysis</p>
                        </div>
                        <Link href="/image-classification">
                            <div className="bg-[#FF52BF] h-56 w-96 rounded-xl text-black flex items-center justify-center flex-col cursor-pointer hover:opacity-90 active:opacity-100">
                                <Image
                                    src="/image classification.png"
                                    width={150}
                                    height={150}
                                    alt="sentiment analysis icon"
                                />
                                <p>Image Classification</p>
                            </div>
                        </Link>
                    </div>
                    <div className="flex justify-center gap-8">
                        <Link href="/object-detection">
                            <div className="bg-[#D887F5] h-56 w-96 rounded-xl text-black flex items-center justify-center flex-col cursor-pointer hover:opacity-90 active:opacity-100">
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
            </div>
        </main>
    );
}
