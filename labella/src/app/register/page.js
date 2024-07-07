'use client'
import React, { useState } from 'react';
import Image from "next/image";
import Link from 'next/link';
import Navbar from '@/components/nav/NavBar';
import InputBox from "@/components/forms/InputBox";
import InputPassword from '@/components/InputPassword';
import axios from 'axios';
import { useRouter } from 'next/navigation'

export default function Register() {
    const apiEndpoint = process.env.NEXT_PUBLIC_API_ENDPOINT + '/user/register';
    const [formData, setFormData] = useState({
        username: '',
        password: '',
        email:''
    });

    const [errors, setErrors] = useState({});
    const [isSubmitting, setIsSubmitting] = useState(false);
    const [alreadyCreated, setAlreadyCreated] = useState(false)
    const handleChange = (e) => {
        const { name, value } = e.target;
        setFormData({
            ...formData,
            [name]: value
        });
    };

    const validate = () => {
        let errors = {};

        if (!formData.username) {
            errors.username = 'Username is required';
        }
        if (!formData.password) {
            errors.password = 'Password is required';
        }
        if (!formData.email) {
            errors.email = 'Email is required';
        }
        return errors;
    };
    const router = useRouter();
    const handleSubmit = async (e) => {
        e.preventDefault();
        const validationErrors = validate();
        setErrors(validationErrors);

        if (Object.keys(validationErrors).length === 0) {
            setIsSubmitting(true);
            try {
                const response = await axios.post(apiEndpoint, {
                    username: formData.username,
                    password: formData.password,
                    email: formData.email
                });
                // Handle successful submission
                console.log("res:",response.status)
                if (response.status == 201) {
                    setAlreadyCreated(false);
                    localStorage.setItem('jwt', response.data.token)
                    router.push("/home")
                }
            } catch (error) {
                console.error('Error submitting form:', error);
                console.log(response)
                if (error.response.status == 409) {
                    setAlreadyCreated(true)
                }
            } finally {
                setIsSubmitting(false);
            }

            setTimeout(() => {
                setIsSubmitting(false);
            }, 1000);
        }

    };


    return (
        <main className="flex flex-col min-h-screen px-24 pb-24 bg-[#19151E] z-20">
            <Navbar />
            <form onSubmit={handleSubmit}>
                <div className="flex flex-col items-center h-screen place-content-center">
                    <p className={`font-bold text-xl mb-4 ${alreadyCreated ? "" : "mb-8"}`}>Create an account</p>
                    {alreadyCreated ? (<div><p className='text-red-500 text-sm'>Email/Username already exists</p></div>) : (<div></div>)}
                    <div className="border border-white border-opacity-50 flex flex-col justify-center items-center rounded p-4 gap-2">
                        <InputBox label={"Email"}
                            name="email"
                            value={formData.email}
                            onChange={handleChange}
                            error={errors.email}
                        />
                        <InputBox label={"Username"}
                            name="username"
                            value={formData.username}
                            onChange={handleChange}
                            error={errors.username}
                        />
                        <InputPassword label={"Password"}
                            name="password"
                            value={formData.password}
                            onChange={handleChange}
                            error={errors.password}
                        />
                        <button type="submit" className="mt-2 flex text-white w-full p-2 rounded-lg h-8 bg-[#FF52BF] justify-center items-center cursor-pointer" disabled={isSubmitting}>
                            {isSubmitting ? 'Signing up...' : 'Sign up'}
                        </button>
                    </div>
                    <div className='mt-4'>
                        <p className='text-sm'>Already on Labella? &nbsp;
                            <Link href="/register"><span className='text-[#ACDFEF] cursor-pointer'>Login.</span></Link>
                        </p>
                    </div>
                </div>
            </form>
        </main>
    );
}
