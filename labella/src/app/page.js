import Image from "next/image";
import Link from 'next/link';
import Navbar from "@/components/NavBar";

export default function Home() {
  return (
    <main className="flex flex-col min-h-screen px-24 pb-24 bg-[#19151E] z-20">
      <Navbar />
      <div className="flex flex-col items-center h-screen place-content-center">
        <p className="text-5xl font-bold mt-32"><span className="text-[#1ED2EC]">Smarter</span> Data Labeling</p>
        <p className="text-5xl font-bold mb-10">Accelerated by <span className="text-[#FFE261]">Active Learning</span></p>
        <div className="bg-[#FF52BF] w-32 h-8 place-content-center text-center rounded-full font-bold cursor-pointer hover:opacity-90 active:opacity-100">
          <Link href="/login"><p>Get started</p></Link>
        </div>
      </div>
    </main>
  );
}
