'use client'
import React, { useState } from 'react';

export default function LabelButton({ classInteger, name, handleOptionChange, bgColour = 'bg-[#FF52BF]', selectedOption}) {
    const label = selectedOption?.labels
    return (
        <div onClick={() => handleOptionChange(classInteger)}>
            <button htmlFor="input-box" className={`mb-2 ${label == classInteger ? "bg-[#FF52BF] text-black" : "bg-black text-white"} border border-white px-4 py-2 rounded-full hover:${bgColour} hover:text-black`}>
                {name}
            </button>
        </div>
    )
}