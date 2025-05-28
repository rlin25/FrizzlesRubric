import React, { useState } from 'react';
import { Box, Grid, AppBar, Toolbar, Typography, Container, Paper, TextField, Button, CircularProgress, Alert, Avatar } from '@mui/material';
import { DragDropContext, Droppable, Draggable } from 'react-beautiful-dnd';
import LightbulbIcon from '@mui/icons-material/Lightbulb';
import DescriptionIcon from '@mui/icons-material/Description';
import ViewModuleIcon from '@mui/icons-material/ViewModule';
import GrainIcon from '@mui/icons-material/Grain';
import BuildIcon from '@mui/icons-material/Build';
import RepeatIcon from '@mui/icons-material/Repeat';

const EXPERTS = [
  { key: 'expert_1_clarity', label: 'Clarity' },
  { key: 'expert_2_documentation', label: 'Documentation' },
  { key: 'expert_3_structure', label: 'Structure' },
  { key: 'expert_4_granulation', label: 'Granulation' },
  { key: 'expert_5_tooling', label: 'Tooling' },
  { key: 'expert_6_repetition', label: 'Repetition' },
];

const EXAMPLE_PROMPTS = [
  "Design an algorithm to efficiently merge overlapping time intervals from multiple calendars",
  "1. Select this Python module. 2. Add type hints to all functions. 3. Use the 'typing' module.",
  "Add proper type checking to the database models in models/index.ts.",
  "Implement error handling in the security module",
  "Create a Towers of Hanoi game using ChatGPT.",
  "Enter this prompt twice."
];

function getLightColor(result) {
  if (result === 1) return '#4caf50'; // green
  if (result === 0) return '#f44336'; // red
  if (result === 'e') return '#ffeb3b'; // yellow
  return '#bdbdbd'; // gray (no result)
}

function App() {
  const [prompt, setPrompt] = useState('');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [results, setResults] = useState(null);
  const [examplePrompts, setExamplePrompts] = useState(EXAMPLE_PROMPTS);

  const handleSubmit = async (e) => {
    e.preventDefault();
    setError('');
    setLoading(true);
    setResults(null);
    try {
      const response = await fetch('/orchestrate', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ prompt })
      });
      if (!response.ok) throw new Error('API error');
      const data = await response.json();
      setResults(data);
    } catch (err) {
      setError('Failed to get results from orchestrator API.');
    } finally {
      setLoading(false);
    }
  };

  // Drag-and-drop logic
  const onDragEnd = (result) => {
    if (!result.destination) return;
    // If dropped in the prompt input area, set the prompt
    if (result.destination.droppableId === 'promptInput') {
      setPrompt(examplePrompts[result.source.index]);
    }
  };

  return (
    <Box sx={{ minHeight: '100vh', display: 'flex', flexDirection: 'column' }}>
      {/* Header */}
      <AppBar position="static">
        <Toolbar>
          <Typography variant="h6" component="div" sx={{ flexGrow: 1 }}>
            Cursor AI Prompt Evaluator
          </Typography>
        </Toolbar>
      </AppBar>

      {/* Main Content */}
      <Container maxWidth="lg" sx={{ flex: 1, py: 4 }}>
        <DragDropContext onDragEnd={onDragEnd}>
          <Grid container spacing={4}>
            {/* Left Column: Prompt input and lights */}
            <Grid item xs={12} md={6}>
              <Paper sx={{ p: 3, mb: 2, display: 'inline-block', minWidth: 0, maxWidth: '100%' }}>
                <Typography variant="h6" gutterBottom>Prompt Input</Typography>
                <Droppable droppableId="promptInput" direction="vertical">
                  {(provided, snapshot) => (
                    <Box
                      component="form"
                      onSubmit={handleSubmit}
                      ref={provided.innerRef}
                      {...provided.droppableProps}
                      sx={{
                        mb: 2,
                        display: 'flex',
                        flexDirection: 'row',
                        gap: 2,
                        alignItems: 'center',
                        background: snapshot.isDraggingOver ? '#e3f2fd' : 'inherit',
                        borderRadius: 1,
                        transition: 'all 0.2s',
                        minHeight: snapshot.isDraggingOver ? 80 : 0,
                        p: snapshot.isDraggingOver ? 1 : 0
                      }}
                    >
                      <TextField
                        label="Enter your prompt"
                        variant="outlined"
                        fullWidth
                        value={prompt}
                        onChange={e => setPrompt(e.target.value)}
                        disabled={loading}
                        multiline
                        minRows={snapshot.isDraggingOver ? 3 : 2}
                        sx={{
                          transition: 'all 0.2s',
                          background: snapshot.isDraggingOver ? '#e3f2fd' : 'inherit',
                        }}
                      />
                      <Button type="submit" variant="contained" disabled={loading || !prompt.trim()} sx={{ height: 56, alignSelf: 'center' }}>
                        {loading ? <CircularProgress size={24} /> : 'Submit'}
                      </Button>
                      {provided.placeholder}
                    </Box>
                  )}
                </Droppable>
                {error && <Alert severity="error" sx={{ mb: 2 }}>{error}</Alert>}
                <Typography variant="h6" gutterBottom sx={{ mt: 2, textAlign: 'left' }}>Expert Results</Typography>
                {/* Visual feedback (lights) */}
                <Box sx={{ display: 'flex', justifyContent: 'flex-start', alignItems: 'center', gap: 3, mb: 1 }}>
                  {EXPERTS.map(expert => (
                    <Box key={expert.key} sx={{ display: 'flex', flexDirection: 'column', alignItems: 'center' }}>
                      <Avatar
                        sx={{
                          bgcolor: getLightColor(results ? results[expert.key] : undefined),
                          width: 40,
                          height: 40,
                          transition: 'background-color 0.3s',
                          border: '2px solid #888',
                          mb: 0.5
                        }}
                      >
                        {expert.key === 'expert_1_clarity' && <LightbulbIcon />}
                        {expert.key === 'expert_2_documentation' && <DescriptionIcon />}
                        {expert.key === 'expert_3_structure' && <ViewModuleIcon />}
                        {expert.key === 'expert_4_granulation' && <GrainIcon />}
                        {expert.key === 'expert_5_tooling' && <BuildIcon />}
                        {expert.key === 'expert_6_repetition' && <RepeatIcon />}
                      </Avatar>
                      <Typography variant="caption" sx={{ textAlign: 'center' }}>{expert.label}</Typography>
                    </Box>
                  ))}
                </Box>
              </Paper>
            </Grid>

            {/* Right Column: Example prompts table with drag-and-drop */}
            <Grid item xs={12} md={6}>
              <Paper sx={{ p: 3, display: 'inline-block', minWidth: 0, maxWidth: '100%' }}>
                <Typography variant="h6" gutterBottom>Example Prompts</Typography>
                <Droppable droppableId="examplePrompts">
                  {(provided, snapshot) => (
                    <Box
                      ref={provided.innerRef}
                      {...provided.droppableProps}
                      sx={{ minHeight: 200, background: snapshot.isDraggingOver ? '#e3f2fd' : '#f5f5f5', borderRadius: 1, p: 2 }}
                    >
                      {examplePrompts.map((promptText, idx) => (
                        <Draggable key={promptText} draggableId={promptText} index={idx}>
                          {(provided, snapshot) => (
                            <Paper
                              ref={provided.innerRef}
                              {...provided.draggableProps}
                              {...provided.dragHandleProps}
                              sx={{ p: 1.5, mb: 1, background: snapshot.isDragging ? '#bbdefb' : 'white', cursor: 'grab', userSelect: 'none' }}
                            >
                              {promptText}
                            </Paper>
                          )}
                        </Draggable>
                      ))}
                      {provided.placeholder}
                    </Box>
                  )}
                </Droppable>
              </Paper>
            </Grid>
          </Grid>
        </DragDropContext>
      </Container>

      {/* Footer */}
      <Box component="footer" sx={{ py: 2, textAlign: 'center', background: '#eee' }}>
        <Typography variant="body2">&copy; 2025 Richard Lin</Typography>
      </Box>
    </Box>
  );
}

export default App; 