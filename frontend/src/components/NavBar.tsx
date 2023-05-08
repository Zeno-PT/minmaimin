import React from 'react';
import { createStyles, makeStyles, Theme } from '@material-ui/core/styles';
import AppBar from '@material-ui/core/AppBar';
import Toolbar from '@material-ui/core/Toolbar';
import Typography from '@material-ui/core/Typography';
import Button from '@material-ui/core/Button';
import IconButton from '@material-ui/core/IconButton';
import MenuIcon from '@material-ui/icons/Menu';

const useStyles = makeStyles((theme: Theme) => 
    createStyles({
        root: {
            flexGrow: 1,
            background: '#ff7b54'
        },
        toolbar: theme.mixins.toolbar,
        menuButton: {
            marginRight: theme.spacing(2),
        },
        title: {
            flexGrow: 1,
            marginLeft: '1.75em'
        },
    })
);

export default function NavBar() {
    const classes = useStyles();

    return (
        <div className={classes.toolbar}>
            <AppBar position="fixed" className={classes.root}>
            <Toolbar>
            <IconButton edge="start" className={classes.menuButton} color="inherit" aria-label="menu">
                <MenuIcon />
            </IconButton>
            <Typography variant="h6" className={classes.title}>
                หมิ่นไม่หมิ่น
            </Typography>
            <Button color="inherit">ลงชื่อเข้าใช้</Button>
            </Toolbar>
        </AppBar>
        </div>
    )
}