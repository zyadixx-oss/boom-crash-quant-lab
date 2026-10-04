import { StrictMode } from 'react';
import { createRoot } from 'react-dom/client';
import App from './App';
import './index.css';

const root = createRoot(document.getElementById('root')!);
root.render(<StrictMode><App /></StrictMode>);
if (import.meta.hot) import.meta.hot.dispose(() => root.unmount());
