import { render, screen } from '@testing-library/react';
import App from './App';

test('renders the video loader', () => {
  render(<App />);
  expect(screen.getByRole('heading', { name: /speedtube/i })).toBeInTheDocument();
  expect(screen.getByPlaceholderText(/paste a youtube url/i)).toBeInTheDocument();
  expect(screen.getByRole('button', { name: /load/i })).toBeDisabled();
});
