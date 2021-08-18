package net.lifove.clami;


public class LabelingException extends Exception{
	
	public LabelingException()
	{
		super("Class Label must be binary");
	}
	
	public LabelingException(String message)
	{
		super(message);
		
	}

}
