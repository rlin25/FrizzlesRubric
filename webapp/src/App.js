import React, { useState } from 'react';
import { Box, Grid, AppBar, Toolbar, Typography, Container, Paper, TextField, Button, CircularProgress, Alert, Avatar, Tooltip } from '@mui/material';
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
  { id: '1', text: "Design an algorithm to efficiently merge overlapping time intervals from multiple calendars" },
  { id: '2', text: "1. Select this Python module. 2. Add type hints to all functions. 3. Use the 'typing' module." },
  { id: '3', text: "Add proper type checking to the database models in models/index.ts." },
  { id: '4', text: "Implement error handling in the security module" },
  { id: '5', text: "Create a Towers of Hanoi game using ChatGPT." },
  { id: '6', text: "Enter this prompt twice." }
];

const EXPERT_TOOLTIPS = {
  expert_1_clarity: `Assesses how clear and understandable the prompt is, ensuring it leaves little room for ambiguity or misinterpretation. Prompts must also be free of spelling and grammar mistakes.`,
  expert_2_documentation: `Evaluates whether the prompt breaks down its primary objective into distinct, individually actionable steps, providing clear guidance for execution.`,
  expert_3_structure: `Checks that any files referenced in the prompt include their absolute or relative paths, ensuring precise identification and preventing confusion during execution.`,
  expert_4_granulation: `Measures the level of detail and specificity in the prompt. Proper granulation ensures the prompt is sufficiently detailed to be actionable, but not overloaded with unnecessary information.`,
  expert_5_tooling: `Determines whether the prompt effectively leverages non-AI tools, libraries, or resources that are relevant to the task, enhancing the quality and efficiency of the solution.`,
  expert_6_repetition: `Detects unnecessary repetition of previously entered prompts, ensuring that each prompt is unique and not redundant.`
};

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
      setPrompt(examplePrompts[result.source.index].text);
    }
  };

  return (
    <Box sx={{ minHeight: '100vh', display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center' }}>
      {/* Header */}
      <AppBar position="static" sx={{ bgcolor: '#f5f5f5', color: '#222' }}>
        <Toolbar>
          <Typography variant="h6" component="div" sx={{ flexGrow: 1, textAlign: 'center' }}>
            Cursor AI Prompt Evaluator
          </Typography>
        </Toolbar>
      </AppBar>

      {/* Main Content */}
      <Container maxWidth="lg" sx={{ flex: 1, py: 4, display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
        <DragDropContext onDragEnd={onDragEnd}>
          <Grid container spacing={4} direction="column" alignItems="center" justifyContent="center">
            {/* Prompt input and lights */}
            <Grid item xs={12} sx={{ display: 'flex', justifyContent: 'center', width: '100%' }}>
              <Paper sx={{ p: 3, mb: 2, display: 'flex', flexDirection: 'column', alignItems: 'center', minWidth: 0, maxWidth: '600px', width: '100%' }}>
                <Typography variant="h6" gutterBottom sx={{ textAlign: 'center' }}>Prompt Input</Typography>
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
                        minHeight: 56,
                        p: snapshot.isDraggingOver ? 2 : 1,
                        boxShadow: snapshot.isDraggingOver ? 4 : 1,
                        border: snapshot.isDraggingOver ? '2px solid #90caf9' : '1px solid #ccc',
                        justifyContent: 'center',
                        width: '100%'
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
                        minRows={2}
                        sx={{
                          transition: 'all 0.2s',
                          background: snapshot.isDraggingOver ? '#e3f2fd' : 'inherit',
                          opacity: snapshot.isDraggingOver ? 0 : 1,
                        }}
                      />
                      <Button type="submit" variant="contained" disabled={loading || !prompt.trim()} sx={{ height: 56, alignSelf: 'center', transition: 'opacity 0.2s', opacity: snapshot.isDraggingOver ? 0 : 1, bgcolor: '#f5f5f5', color: '#222', '&:hover': { bgcolor: '#e0e0e0' } }}>
                        {loading ? <CircularProgress size={24} /> : 'Submit'}
                      </Button>
                      {provided.placeholder}
                    </Box>
                  )}
                </Droppable>
                {error && <Alert severity="error" sx={{ mb: 2 }}>{error}</Alert>}
                <Typography variant="h6" gutterBottom sx={{ mt: 2, textAlign: 'center' }}>Expert Results</Typography>
                {/* Visual feedback (lights) */}
                <Box
                  sx={{
                    display: 'grid',
                    gridTemplateColumns: { xs: 'repeat(3, 1fr)', sm: 'repeat(6, 1fr)' },
                    gap: 3,
                    mb: 1,
                    justifyItems: 'center',
                    alignItems: 'center',
                  }}
                >
                  {EXPERTS.map(expert => (
                    <Tooltip key={expert.key} title={EXPERT_TOOLTIPS[expert.key]} arrow placement="bottom" 
                      sx={{
                        '& .MuiTooltip-tooltip': {
                          fontSize: '1.2rem',
                        }
                      }}
                    >
                      <Box sx={{ display: 'flex', flexDirection: 'column', alignItems: 'center' }}>
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
                    </Tooltip>
                  ))}
                </Box>
              </Paper>
            </Grid>

            {/* Example prompts table with drag-and-drop */}
            <Grid item xs={12} sx={{ display: 'flex', justifyContent: 'center', width: '100%' }}>
              <Paper sx={{ p: 3, display: 'flex', flexDirection: 'column', alignItems: 'center', minWidth: 0, maxWidth: '600px', width: '100%' }}>
                <Typography variant="h6" gutterBottom sx={{ textAlign: 'center' }}>Example Prompts</Typography>
                <Droppable droppableId="examplePrompts">
                  {(provided, snapshot) => (
                    <Box
                      ref={provided.innerRef}
                      {...provided.droppableProps}
                      sx={{ minHeight: 200, background: snapshot.isDraggingOver ? '#e3f2fd' : '#f5f5f5', borderRadius: 1, p: 2, display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', width: '100%' }}
                    >
                      {examplePrompts.map((promptObj, idx) => (
                        <Draggable key={promptObj.id} draggableId={promptObj.id} index={idx}>
                          {(provided, snapshot) => (
                            <Paper
                              ref={provided.innerRef}
                              {...provided.draggableProps}
                              {...provided.dragHandleProps}
                              sx={{ p: 1.5, mb: 1, background: snapshot.isDragging ? '#bbdefb' : 'white', cursor: 'grab', userSelect: 'none', textAlign: 'center', width: '100%' }}
                            >
                              {promptObj.text}
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
      <Box component="footer" sx={{ py: 2, textAlign: 'center', background: '#eee', width: '100%' }}>
        <Typography variant="body2">&copy; 2025 Richard Lin</Typography>
      </Box>
    </Box>
  );
}

export default App; 