'use client'
import React, { useState } from 'react';

export default function LabelButton({ classInteger, name, handleOptionChange }) {
    return (
        <div onClick={() => handleOptionChange(classInteger)}>
            <button htmlFor="input-box" className="mb-2 text-white bg-black border border-white px-4 py-2 rounded-full hover:bg-[#FF52BF] hover:text-black">
                {name}
            </button>
        </div>
    )
}