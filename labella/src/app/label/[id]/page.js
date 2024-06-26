'use client';
import React, { useState, useEffect } from 'react';
import Image from "next/image";
import Link from 'next/link';
import LabelButton from '@/components/LabelButton';
import ImageSlider from '@/components/ImageSlider';
import SideNav from '@/components/SideNav';
import axios from 'axios';
import { useRouter } from 'next/navigation';

export default function Label({ params }) {
    const apiEndpoint = process.env.NEXT_PUBLIC_API_ENDPOINT + `/dataset/${params.id}/batch`;
    const jwtToken = localStorage.getItem('jwt');

    const config = {
        headers: {
            'Content-Type': 'application/json',
            'Authorization': `Bearer ${jwtToken}`
        }
    };

    const [images, setImages] = useState([]);
    const [error, setError] = useState(null);
    const [loading, setLoading] = useState(true);
    const [selectedImage, setSelectedImage] = useState('');

    useEffect(() => {
        const fetchData = async () => {
            try {
                const response = await axios.get(apiEndpoint);
                setImages(response.data);
            } catch (error) {
                setError(error.message);
            } finally {
                setLoading(false);
            }
        };
        fetchData();
    }, [apiEndpoint]); 

    useEffect(() => {
        if (images.length > 0) {
          setSelectedImage(images[0].data); // Set selected image only when data is available
        }
      }, [images]);

    if (loading) {
        return <div>Loading...</div>;
    }
    if (error) {
        return <div>Error: {error}</div>;
    }

    const handleImageChange = (idx) =>{
        setSelectedImage(images[idx].data);
    };

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
                <SideNav params={params.id}/>
                <div className="bg-white border-2 mt-32 ml-64 h-100% hidden lg:block"></div>
                <div className="ml-0 lg:ml-20">
                    <p className="text-xl text-[#FF52BF] font-bold mb-8 mt-40">Image Classification</p>
                    <p className="font-bold mb-2">Label Images</p>
                    <div className='flex gap-4 mb-4 items-center justify-center'>
                        <ImageSlider images={images} bucketname={"dltechjam"} bucketprefix={"transfer-antsbees"} handleImageChange={handleImageChange}/>
                        <div className='bg-[#3B3840] rounded-lg w-96 p-4'>
                            <p className='text-white font-bold mb-2'>Class</p>
                            <div className='flex gap-4'>
                                <LabelButton name="Bee"/>
                                <LabelButton name="Ant"/>
                            </div>
                        </div>
                    </div>
                    <div className='grid grid-cols-10 gap-2'>
                        {images.map((item, index) => {
                            return (
                                <div key={item.data}> <img src={`https://dltechjam.s3.amazonaws.com/transfer-antsbees/${item.data}`} className={`w-20 h-20 rounded-lg ${selectedImage === item.data ? 'border-4 border-white' : ''}`} />
                                    <p className='break-all' style={{'fontSize': '10px'}}>{item.data}</p>
                                </div>
                            )
                        })}
                    </div>
                </div>
            </div>
        </main>
    );
}
