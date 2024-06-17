import Image from "next/image";
import Link from 'next/link';

export default function ImageClassification() {
    const menuOptions = ["Model", "Details and Dataset", "Label", "Fine Tune", "Performance and Statistics"]
    const models = ["model 1", "model 2", "model 3", "model 4", "model 5", "model 6"]
    return (
        <main className="flex flex-col min-h-screen px-24 pb-24 bg-[#19151E] z-20">
            <div className="flex flex-row fixed top-0 h-24 w-full z-20 bg-[#19151E] items-end">
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
                <div className="h-screen bg-white border-2 mt-10 ml-64"></div>
                <div className="ml-20">
                    <p className="text-xl text-[#FF52BF] font-bold my-10">Image Classification</p>
                    <p className="font-bold mb-10">Select Model</p>
                    <div className="grid grid-cols-4 gap-8">
                        {models.map((model, index) => (
                            <div key={index} className="flex border w-64 items-center justify-center rounded-lg h-10 hover:bg-[#FF52BF] cursor-pointer hover:text-black">
                                {model}
                            </div>
                        ))}
                    </div>
                    <p className="font-bold mt-10 mb-4">Provide details for your data labelling</p>
                    <div className="flex flex-col gap-4">
                        <div>
                            <p className="mb-2">Selected Model</p>

                            <div className="flex border w-64 rounded-full pl-4 py-1">Model 1</div>
                        </div>
                        <div>
                            <p className="mb-2">Choose Dataset</p>
                            <div className="flex border w-64 rounded-full pl-4 py-1 items-center">
                                <img src="/upload.png" alt="upload icon" className="h-4 mr-2"></img>
                                <p>Upload Dataset</p>
                            </div>
                        </div>
                        <div>
                            <p className="mb-2">Number of classes</p>
                            <div className="flex border w-64 rounded-full pl-4 py-1 items-center">
                                10
                            </div>
                        </div>
                        <div>
                            <div className="flex bg-[#FF52BF] w-20 rounded-full justify-center items-center cursor-pointer">
                                <p>Label</p>
                            </div>
                        </div>

                    </div>

                </div>
            </div>


        </main>
    );
}
