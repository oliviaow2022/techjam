'use client'
import React, { useState } from 'react';
import Link from 'next/link';
import Navbar from '@/components/nav/NavBar';
import InputBox from "@/components/forms/InputBox";
import InputPassword from '@/components/InputPassword';
import axios from 'axios';
import { useRouter } from 'next/navigation'
import { setJwtToken, setUserId } from '@/store/authSlice';
import { useDispatch } from "react-redux"

export default function Login() {
    const router = useRouter();
    const dispatch = useDispatch(); 

    const apiEndpoint = process.env.NEXT_PUBLIC_API_ENDPOINT + '/user/login';
    const [formData, setFormData] = useState({
        username: '',
        password: ''
    });

    const [errors, setErrors] = useState({});
    const [isSubmitting, setIsSubmitting] = useState(false);
    const [wrongPW, setWrongPW] = useState(false)
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
        return errors;
    };

    const handleSubmit = async (e) => {
        e.preventDefault();
        const validationErrors = validate();
        setErrors(validationErrors);

        if (Object.keys(validationErrors).length === 0) {
            setIsSubmitting(true);
            try {
                const response = await axios.post(apiEndpoint, {
                    username: formData.username,
                    password: formData.password
                });
                // Handle successful submission
                if(response.status == 200) {
                    setWrongPW(false);
                    console.log(response.data.user_id, response.data.token)
                    dispatch(setUserId(response.data.user_id))
                    dispatch(setJwtToken(response.data.token))
                    router.push("/home")
                }
            } catch (error) {
                console.error('Error submitting form:', error);
                if(error.response.status == 401){
                    setWrongPW(true)
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
                    <p className={`font-bold text-xl mb-4 ${wrongPW ? "" : "mb-8"}`}>Sign in to Labella</p>
                    {wrongPW ? (<div><p className='text-red-500 text-sm'>Incorrect username or password</p></div>):(<div></div>)}
                    <div className="border border-white border-opacity-50 flex flex-col justify-center items-center rounded p-4 gap-2">
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
                            {isSubmitting ? 'Logging in...' : 'Login'}
                        </button>
                    </div>
                    <div className='mt-4'>
                        <p className='text-sm'>New to Labella? &nbsp;
                            <Link href="/register"><span className='text-[#ACDFEF] cursor-pointer'>Create an account.</span></Link>
                        </p>
                    </div>
                </div>
            </form>
        </main>
    );
}
