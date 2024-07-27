'use client'
import React, { useState } from 'react';

export default function LabelButton({classInteger, name, handleOptionChange, bgColour = 'bg-[#FF52BF]', selectedOption, isMulticlass}) {
    const label = selectedOption?.labels
    let match = false
    if (isMulticlass){
        const list = classInteger.split("_")
        const categoryIndex = parseInt(list[0])
        const categoryInteger = parseInt(list[1])
        if(label != null && (parseInt(label[categoryIndex]) === categoryInteger)){
            match = true
        }
    } else if(label == classInteger){
        match = true
    }

    return (
        <div onClick={() => handleOptionChange(classInteger)}>
            <button htmlFor="input-box" className={`mb-2 ${match ? "bg-[#FF52BF] text-black" : "bg-black text-white"} border border-white px-4 py-2 rounded-full hover:${bgColour} hover:text-black`}>
                {name}
            </button>
        </div>
    )
}