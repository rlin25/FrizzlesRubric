if __name__ == "__main__":
    import uvicorn
    uvicorn.run("orchestrator_2_0_prompt_quality.orchestrator_api:app", host="0.0.0.0", port=8008, reload=False) 