import React, { Component } from 'react';
import './App.css';
import { Grid, Fab, Slide, Typography } from '@material-ui/core';
import ArrowForwardIosIcon from '@material-ui/icons/ArrowForwardIos';
import { createMuiTheme } from '@material-ui/core/styles';
import { ThemeProvider } from '@material-ui/styles';
import LaunchScreenPatient from './components/LaunchScreenPatient';
import LaunchScreenPhysician from './components/LaunchScreenPhysician';
import axios from 'axios';

const font = "'Montserrat', sans-serif";

const muiTheme = createMuiTheme({
    typography : {
        fontFamily: font
    }
})

class App extends Component {

    componentWillMount() {
        this.setState({
            startPatient: false,
            startPhysician: false
        });
        axios.get('Splash');
    };

    handlePatient() {
        this.setState({
            startPatient: true
        });
    };

    handlePhysician() {
        this.setState({
            startPhysician: true
        });
    }

    render() {
        return (
            <ThemeProvider theme={muiTheme}>
            <div className="App">
                <Grid container>
                    
                </Grid>
            </div>
            </ThemeProvider>
        );
    };
}

export default App;
