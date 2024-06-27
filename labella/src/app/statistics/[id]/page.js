'use client';
import React, { useState, useEffect } from 'react';
import Link from 'next/link';
import SideNav from '@/components/SideNav';
import axios from 'axios';

export default function Statistics ({ params }) {
    const apiEndpoint = process.env.NEXT_PUBLIC_API_ENDPOINT + `/history/${params.id}/info`;
    const jwtToken = localStorage.getItem('jwt');
    const config = {
        headers: {
            'Content-Type': 'application/json',
            'Authorization': `Bearer ${jwtToken}`
        }
    };
    const [history, setHistory] = useState('');
    useEffect(() => {
        const fetchData = async () => {
            try {
                const response = await axios.get(apiEndpoint);
                setHistory(response.data);
            } catch (error) {
                console.error(error.message);
            }
        };
        fetchData();
    }, [apiEndpoint]);
    console.log(history); 
    return(<main className="flex flex-col min-h-screen px-24 pb-24 bg-[#19151E] z-20">
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
            <SideNav params={params.id}/>
            <div className="bg-white border-2 mt-32 ml-64 h-100% hidden lg:block"></div>
            <div className="ml-0 lg:ml-20">
                <p className="text-xl text-[#FF52BF] font-bold mb-8 mt-40">Image Classification</p>
                <p className="font-bold mb-2">Statistics</p>
            </div>
        </div>
    </main>)
}