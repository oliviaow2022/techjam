const FileInput = ({label, error, handleFileChange}) => {
  return (
    <div>
      <label htmlFor="input-zip-file" className="block text-white my-1">
        {label}
      </label>
      <input
        type="file"
        id="input-zip-file"
        accept=".zip"
        onChange={handleFileChange}
      />
      {error && (
        <p className="text-red-500 text-sm">{error}</p>
      )}
    </div>
  );
};

export default FileInput;
