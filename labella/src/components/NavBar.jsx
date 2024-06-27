"use client";

import Image from "next/image";
import Link from "next/link";

export default function Navbar() {
  const isLoggedIn = () => {
    const token = localStorage.getItem("jwt");
    return token !== null;
  };
  return (
    <div className="flex flex-row fixed top-0 h-24 w-10/12 2xl:w-full z-20 bg-[#19151E] items-end">
      <div className="z-10 max-w-5xl w-full justify-between text-sm lg:flex">
        <Link href="/">
          <p className="text-xl font-bold">Labella</p>
        </Link>
      </div>
      <div className="flex justify-around w-96 items-center">
        <p className="mx-4 cursor-pointer">Platform</p>
        <p className="mr-2 cursor-pointer">Datasets</p>
        <p className="mr-2 cursor-pointer">Documentation</p>
        {isLoggedIn && (
          <Image
            src="/profileicon.png"
            width={30}
            height={15}
            alt="profile icon"
            className="cursor-pointer"
          />
        )}
      </div>
    </div>
  );
}
