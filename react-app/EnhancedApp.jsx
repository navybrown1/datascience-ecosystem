import React, { useState, useEffect, useMemo } from 'react';
import './App.css';

// Simplified version of the growth tracking app with a measurement
// history table that allows deleting entries.

const MeasurementTable = ({ measurements, onDelete }) => (
  <table className="table-auto w-full text-left mt-4">
    <thead>
      <tr>
        <th className="px-2">Date</th>
        <th className="px-2">Weight (kg)</th>
        <th className="px-2">Length (cm)</th>
        <th className="px-2"></th>
      </tr>
    </thead>
    <tbody>
      {measurements.map((m, idx) => (
        <tr key={idx} className="border-t">
          <td className="px-2 py-1">{m.date}</td>
          <td className="px-2 py-1">{m.weight}</td>
          <td className="px-2 py-1">{m.length}</td>
          <td className="px-2 py-1 text-right">
            <button
              className="text-red-600 hover:underline"
              onClick={() => onDelete(idx)}
            >
              Delete
            </button>
          </td>
        </tr>
      ))}
    </tbody>
  </table>
);

const AddMeasurement = ({ onAdd }) => {
  const [form, setForm] = useState({
    date: new Date().toISOString().split('T')[0],
    weight: '',
    length: ''
  });
  const isValid = form.date && form.weight && form.length;

  return (
    <div className="flex gap-2 mt-4">
      <input
        type="date"
        className="border p-1 rounded"
        value={form.date}
        onChange={e => setForm({ ...form, date: e.target.value })}
      />
      <input
        type="number"
        placeholder="weight"
        className="border p-1 rounded w-24"
        value={form.weight}
        onChange={e => setForm({ ...form, weight: e.target.value })}
      />
      <input
        type="number"
        placeholder="length"
        className="border p-1 rounded w-24"
        value={form.length}
        onChange={e => setForm({ ...form, length: e.target.value })}
      />
      <button
        disabled={!isValid}
        className="bg-blue-600 text-white px-3 rounded disabled:opacity-50"
        onClick={() => {
          onAdd({
            date: form.date,
            weight: parseFloat(form.weight),
            length: parseFloat(form.length)
          });
          setForm({ date: form.date, weight: '', length: '' });
        }}
      >
        Add
      </button>
    </div>
  );
};

export default function EnhancedApp() {
  const [measurements, setMeasurements] = useState([]);

  const handleAdd = m => {
    setMeasurements(prev => [...prev, m]);
  };

  const handleDelete = idx => {
    setMeasurements(prev => prev.filter((_, i) => i !== idx));
  };

  return (
    <div className="p-4 max-w-xl mx-auto">
      <h1 className="text-2xl font-bold mb-2">Growth Tracker</h1>
      <AddMeasurement onAdd={handleAdd} />
      {measurements.length > 0 && (
        <MeasurementTable measurements={measurements} onDelete={handleDelete} />
      )}
    </div>
  );
}
