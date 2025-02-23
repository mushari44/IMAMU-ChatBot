import React, { useState } from "react";
import UserIcon from "../assets/user.png";
import editIcon from "../assets/edit-text.svg";

const UserReq = ({ query, timestamp, onUpdate }) => {
  const [editing, setEditing] = useState(false);
  const [editedMessage, setEditedMessage] = useState(query);

  const handleEdit = () => {
    setEditing(true);
  };

  const handleSaveEdit = () => {
    if (!editedMessage.trim()) return;
    onUpdate(editedMessage);
    setEditing(false);
  };

  const handleCancelEdit = () => {
    setEditedMessage(query);
    setEditing(false);
  };

  return (
    <div
      className={`rounded-md mb-2 User-Chat ${editing ? "bg-yellow-100" : ""}`}
    >
      <div className="Time-Icon flex items-center space-x-2">
        <img src={UserIcon} alt="User" className="w-8 h-8" />
        <p className="text-sm">{timestamp}</p>
        {!editing && (
          <button
            aria-label="Edit message"
            className="Edit-Button"
            onClick={handleEdit}
          >
            <img src={editIcon} alt="Edit" className="Edit-Icon" />
          </button>
        )}
      </div>

      <div>
        {editing ? (
          <div className="flex items-center">
            <input
              type="text"
              className="User-Query p-2 border rounded-md flex-1"
              value={editedMessage}
              onChange={(e) => setEditedMessage(e.target.value)}
            />
            <button
              onClick={handleSaveEdit}
              className="ml-2 text-green-500 hover:text-green-600"
            >
              Save
            </button>
            <button
              onClick={handleCancelEdit}
              className="ml-2 text-red-500 hover:text-red-600"
            >
              Cancel
            </button>
          </div>
        ) : (
          <p className="User-Query">{query}</p>
        )}
      </div>
    </div>
  );
};

export default UserReq;
