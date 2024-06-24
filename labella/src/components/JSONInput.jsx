'use client';
import React, {useState} from 'react';
import TextArea from "./TextArea";

const JsonInput = ({label, name, onJsonChange}) => {
    const [jsonString, setJsonString] = useState('');
    const [jsonObject, setJsonObject] = useState(null);
    const [error, setError] = useState(null);

    const handleChange = (e) => {
        const newValue = e.target.value;
        setJsonString(newValue);
        try {
            // Validate the JSON string
            const parsedJson = JSON.parse(newValue);
            setJsonObject(parsedJson);
            setError(null);
            console.log("Parsed JSON:", parsedJson);
            onJsonChange(parsedJson);
        } catch (e) {
            setError('Invalid JSON. Please ensure it is properly formatted.');
            console.log("Invalid JSON");
            setJsonObject(null);
        }
    };

    return (
        <div className="">
            <TextArea
                label={label}
                name={name}
                value={jsonString}
                onChange={handleChange}
                error={error}
            />
            {jsonObject && (
                <div className="mt-4">
                    <h3 className="">Parsed JSON Object:</h3>
                    <pre className="bg-gray-100 p-2 rounded-lg text-black">{JSON.stringify(jsonObject, null, 2)}</pre>
                </div>
            )}
        </div>
    );
};

export default JsonInput;