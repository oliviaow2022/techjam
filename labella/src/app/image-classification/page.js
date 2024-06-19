import Image from "next/image";
import Link from 'next/link';
import InputBox from "@/components/InputBox";

export default function ImageClassification() {
    const menuOptions = ["Project Details", "Model", "Dataset", "Label", "Fine Tune", "Performance and Statistics"]
    const models = ["Resnet18", "Densenet121", "Alexnet", "Convnext_base"]
    return (
        <main className="flex flex-col min-h-screen px-24 pb-24 bg-[#19151E] z-20">
            <div className="flex flex-row fixed top-0 h-24 w-10/12 2xl:w-full z-20 bg-[#19151E] items-end">
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
                <div className="bg-white border-2 mt-32 ml-64 h-screen sticky"></div>
                <div className="ml-20">
                    <p className="text-xl text-[#FF52BF] font-bold mb-8 mt-40">Image Classification</p>
                    <div className="mb-4">
                        <p className="font-bold mb-4">Create a new project</p>
                        <InputBox label={"Project Name"} />
                    </div>

                    <div className="mb-4">
                        <p className="mt-4">Select Model</p>
                        <div className="grid grid-cols-4 gap-6">
                            {models.map((model, index) => (
                                <div key={index} className="flex border w-40 2xl:w-64 items-center justify-center rounded-lg h-8 hover:bg-[#FF52BF] cursor-pointer hover:text-black">
                                    {model}
                                </div>
                            ))}
                        </div>
                    </div>

                    <div className="mb-4">
                        <InputBox label={"Number of training epochs"} />
                    </div>

                    <div className="mb-4">
                        <InputBox label={"Train-test split ratio"} />
                    </div>

                    <div className="mb-4">
                        <InputBox label={"Batch size"} />
                    </div>


                    <div className="mb-4">
                        <p className="">Choose Dataset</p>
                        <div className="flex border w-64 rounded-lg pl-4 py-1 items-center">
                            <img src="/upload.png" alt="upload icon" className="h-4 mr-2"></img>
                            <p>Upload Dataset</p>
                        </div>
                    </div>
                    <div className="mb-4">
                        <p className="mb-2">Number of classes</p>
                        <div className="flex border w-64 rounded-lg pl-4 py-1 items-center">
                            10
                        </div>
                    </div>
                    <div>
                        <div className="flex bg-[#FF52BF] w-32 rounded-full justify-center items-center cursor-pointer">
                            <p>Create Project</p>
                        </div>
                    </div>



                </div>
            </div>


        </main>
    );
}
