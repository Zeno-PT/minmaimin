import React, { useState, useEffect } from 'react';
import { makeStyles, Theme, createStyles } from '@material-ui/core/styles';
import CircularProgress from '@material-ui/core/CircularProgress';
import Backdrop from '@material-ui/core/Backdrop';
import Paper from '@material-ui/core/Paper';

const useStyles = makeStyles((theme: Theme) => 
    createStyles({
        root: {
            display: 'flex',
            '& > * + *': {
                marginLeft: theme.spacing(2),
            }
        },
        backdrop: {
            zIndex: theme.zIndex.drawer + 1,
            color: '#fff'
        },
        paper: {
            maxWidth: '200',
            maxHeight: '200'
        }
    })
);

export default function LoadingModal(handle) {
    const classes = useStyles();
    const [progress, setProgress] = useState(0);
    const [open, setOpen] = useState(false);
    const [isMount, setIsMount] = useState(true);

    const handleClose = () => {
        setOpen(false);
    };

    const handleToggle = () => {
        setOpen(!open);
    };
    
    function startTimer() {
        return setInterval(() => {
            setProgress((prevProgress) => {
                if (prevProgress === 153) {
                    setOpen(false);
                }
                return prevProgress + 1;
            });
        }, 25);
    }

    useEffect(() => {
        if (isMount) {
            setIsMount(false);
            return;
        }
        const timer = startTimer();

        handleToggle();
    }, [handle.props]);

    return (
        <div className={classes.root}>
            <Backdrop className={classes.backdrop} open={open} onClick={handleClose}>
                <Paper elevation={24}>
                    <CircularProgress variant='determinate' value={progress} style={{margin: '30px'}} size={100}/>
                </Paper>
            </Backdrop>
        </div>
    );
}