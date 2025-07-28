import React, { useState } from "react";
import * as XLSX from "xlsx";

const IDSFrontend = () => {
  const [formData, setFormData] = useState({
    protocol_type: "tcp",
    service: "http",
    flag: "SF",
    src_bytes: "",
    dst_bytes: "",
    count: "",
    srv_count: "",
  });

  const [singleResult, setSingleResult] = useState(null);
  const [bulkResults, setBulkResults] = useState([]);
  const [file, setFile] = useState(null);
  const [loading, setLoading] = useState(false);

  const API_URL = "http://127.0.0.1:5000/predict";

  const handleChange = (e) => {
    setFormData({ ...formData, [e.target.name]: e.target.value });
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    setLoading(true);
    try {
      const response = await fetch(API_URL, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(formData),
      });
      const result = await response.json();
      setSingleResult(result);
    } catch (error) {
      console.error("Prediction error:", error);
    }
    setLoading(false);
  };

  const handleFileChange = (e) => {
    setFile(e.target.files[0]);
  };

  const handleFileUpload = async () => {
    if (!file) {
      alert("Please select a file");
      return;
    }
    setLoading(true);
    const reader = new FileReader();
    reader.onload = async (e) => {
      const data = new Uint8Array(e.target.result);
      const workbook = XLSX.read(data, { type: "array" });
      const sheetName = workbook.SheetNames[0];
      const sheet = XLSX.utils.sheet_to_json(workbook.Sheets[sheetName]);

      const predictions = [];
      for (const row of sheet) {
        try {
          const response = await fetch(API_URL, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(row),
          });
          const res = await response.json();
          predictions.push({ input: row, prediction: res });
        } catch (err) {
          console.error("Batch prediction error:", err);
        }
      }
      setBulkResults(predictions);
      setLoading(false);
    };
    reader.readAsArrayBuffer(file);
  };

  const riskColor = (risk) => {
    if (risk === "LOW") return "text-green-400";
    if (risk === "MEDIUM") return "text-yellow-400";
    return "text-red-500";
  };

  return (
    <div className="p-6 max-w-4xl mx-auto">
      <h2 className="text-3xl font-bold mb-6 text-center text-purple-400">
        Intrusion Detection System
      </h2>

      {/* Manual Input Form */}
      <form
        className="grid grid-cols-2 gap-4 bg-gray-900 p-6 rounded-lg shadow-lg"
        onSubmit={handleSubmit}
      >
        <label className="flex flex-col">
          Protocol Type:
          <select
            name="protocol_type"
            value={formData.protocol_type}
            onChange={handleChange}
            className="p-2 mt-1 rounded bg-gray-800 text-white border border-gray-600"
          >
            <option value="tcp">TCP</option>
            <option value="udp">UDP</option>
            <option value="icmp">ICMP</option>
          </select>
        </label>

        <label className="flex flex-col">
          Service:
          <select
            name="service"
            value={formData.service}
            onChange={handleChange}
            className="p-2 mt-1 rounded bg-gray-800 text-white border border-gray-600"
          >
            <option value="http">HTTP</option>
            <option value="ftp">FTP</option>
            <option value="smtp">SMTP</option>
          </select>
        </label>

        <label className="flex flex-col">
          Flag:
          <select
            name="flag"
            value={formData.flag}
            onChange={handleChange}
            className="p-2 mt-1 rounded bg-gray-800 text-white border border-gray-600"
          >
            <option value="SF">SF</option>
            <option value="REJ">REJ</option>
            <option value="S0">S0</option>
          </select>
        </label>

        <label className="flex flex-col">
          Source Bytes:
          <input
            type="number"
            name="src_bytes"
            value={formData.src_bytes}
            onChange={handleChange}
            className="p-2 mt-1 rounded bg-gray-800 text-white border border-gray-600"
          />
        </label>

        <label className="flex flex-col">
          Destination Bytes:
          <input
            type="number"
            name="dst_bytes"
            value={formData.dst_bytes}
            onChange={handleChange}
            className="p-2 mt-1 rounded bg-gray-800 text-white border border-gray-600"
          />
        </label>

        <label className="flex flex-col">
          Count:
          <input
            type="number"
            name="count"
            value={formData.count}
            onChange={handleChange}
            className="p-2 mt-1 rounded bg-gray-800 text-white border border-gray-600"
          />
        </label>

        <label className="flex flex-col">
          Service Count:
          <input
            type="number"
            name="srv_count"
            value={formData.srv_count}
            onChange={handleChange}
            className="p-2 mt-1 rounded bg-gray-800 text-white border border-gray-600"
          />
        </label>

        <button
          type="submit"
          className="col-span-2 bg-gradient-to-r from-blue-500 to-purple-600 text-white py-2 rounded-lg hover:scale-105 transition-transform"
        >
          {loading ? "Predicting..." : "Predict"}
        </button>
      </form>

      {/* Single Prediction Result */}
      {singleResult && (
        <div className="bg-gray-800 mt-6 p-4 rounded shadow-lg">
          <h3 className="font-bold text-purple-300">Prediction Result:</h3>
          <p className={riskColor(singleResult.risk_level)}>
            {JSON.stringify(singleResult, null, 2)}
          </p>
        </div>
      )}

      <hr className="my-8 border-gray-700" />

      {/* Bulk Upload */}
      <div className="bg-gray-900 p-4 rounded-lg shadow-lg">
        <h3 className="font-bold mb-2 text-purple-400">
          Upload CSV/Excel for Batch Prediction
        </h3>
        <input
          type="file"
          accept=".csv,.xlsx"
          onChange={handleFileChange}
          className="mb-2 text-white"
        />
        <button
          onClick={handleFileUpload}
          className="ml-2 bg-gradient-to-r from-green-500 to-teal-500 text-white py-1 px-3 rounded hover:scale-105 transition-transform"
        >
          {loading ? "Processing..." : "Upload & Predict"}
        </button>
      </div>

      {/* Bulk Results */}
      {bulkResults.length > 0 && (
        <div className="mt-4">
          <h3 className="font-bold mb-2 text-purple-300">Batch Results:</h3>
          <table className="w-full border border-gray-700 text-sm">
            <thead>
              <tr className="bg-gray-800 text-white">
                <th className="border border-gray-700 p-2">Input</th>
                <th className="border border-gray-700 p-2">Prediction</th>
              </tr>
            </thead>
            <tbody>
              {bulkResults.map((res, index) => (
                <tr key={index} className="hover:bg-gray-700">
                  <td className="border border-gray-700 p-2 text-gray-300">
                    {JSON.stringify(res.input)}
                  </td>
                  <td
                    className={`border border-gray-700 p-2 ${riskColor(
                      res.prediction.risk_level
                    )}`}
                  >
                    {JSON.stringify(res.prediction)}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}

      <hr className="my-8 border-gray-700" />
    </div>
  );
};

export default IDSFrontend;
