import React, { Component } from 'react';
import './App.css';
import { Grid, Fab, Slide, Typography } from '@material-ui/core';
import AddPhotoAlternateIcon from "@material-ui/icons/AddPhotoAlternate";
import { createMuiTheme } from '@material-ui/core/styles';
import { ThemeProvider } from '@material-ui/styles';
import axios from 'axios';
import {DropzoneArea} from 'material-ui-dropzone'

const font = "'Montserrat', sans-serif";

const muiTheme = createMuiTheme({
    typography : {
        fontFamily: font
    }
})

class App extends Component {

    componentWillMount() {
        this.setState({
            imageUploaded: 0
        })
    };

    handleUploadClick = event => {
        console.log(event.target.files.length);
        var file = event.target.files[0];
        const reader = new FileReader();
        var url = reader.readAsDataURL(file);
    
        reader.onloadend = function(e) {
          this.setState({
            selectedFile: [reader.result]
          });
        }.bind(this);
    
        this.setState({
          selectedFile: event.target.files[0],
          imageUploaded: 1
        });
    };

    render() {
        return (
            <ThemeProvider theme={muiTheme}>
            <div className="App">
                <Grid className="App-header" container>
                <input
                    accept="image/*"
                    // className={classes.input}
                    id="contained-button-file"
                    multiple
                    hidden
                    type="file"
                    onChange={this.handleUploadClick}
                />
                <label htmlFor="contained-button-file">
                    <Fab component="span">
                        <AddPhotoAlternateIcon />
                    </Fab>
                </label>
                </Grid>
            </div>
            </ThemeProvider>
        );
    };
}

export default App;
