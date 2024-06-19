import Image from "next/image";
import Link from 'next/link';

export default function Home() {
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
      <div className="flex flex-col items-center h-96 place-content-center">
        <p className="text-5xl font-bold mt-32"><span className="text-[#1ED2EC]">Smarter</span> Data Labeling</p>
        <p className="text-5xl font-bold mb-10">Accelerated by <span className="text-[#FFE261]">Active Learning</span></p>
        <div className="bg-[#FF52BF] w-32 h-8 place-content-center text-center rounded-full font-bold cursor-pointer hover:opacity-90 active:opacity-100">
          <Link href="/home"><p>Get started</p></Link>
        </div>
      </div>
    </main>
  );
}
